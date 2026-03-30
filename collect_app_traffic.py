"""
NetSentinel - App Traffic Data Collector v2
Uses SLIDING WINDOW approach for 10-50x more training samples.

Usage:
    sudo python collect_app_traffic.py --interactive
    sudo python collect_app_traffic.py --app youtube --duration 300
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

try:
    from scapy.all import sniff, IP, TCP, UDP, Ether, Raw, DNS, IPv6
except ImportError:
    print("ERROR: Scapy not installed. Run: pip install scapy")
    sys.exit(1)

import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data_v2", "app_traffic")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SIZE = 2.0
WINDOW_STEP = 0.5
MIN_PACKETS_PER_WINDOW = 3
PSS_LENGTH = 50


class WindowCollector:
    def __init__(self, app_name, interface="en0"):
        self.app_name = app_name
        self.interface = interface
        self.raw_packets = []
        self._our_ips = set()
        self._local_ipv6 = None
        self._detect_local_info()

    def _get_direction(self, pkt):
        # IPv4
        if pkt.haslayer(IP):
            src = pkt[IP].src
            if src.startswith("192.168.") or src.startswith("10.") or src.startswith("172.16"):
                return 1
            return -1
        # IPv6 — check if source is our local IPv6 address
        if pkt.haslayer(IPv6):
            src = pkt[IPv6].src
            # Link-local (fe80::), ULA (fd/fc), or if it matches our known prefix
            if src.startswith("fe80:") or src.startswith("fd") or src.startswith("fc"):
                return 1
            # Check if source is in our local IPv6 prefix (first 64 bits match)
            if hasattr(self, '_local_ipv6') and self._local_ipv6:
                local_prefix = self._local_ipv6.split(":")[:4]
                src_prefix = src.split(":")[:4]
                if local_prefix == src_prefix:
                    return 1
            # Heuristic: if we've seen this src as our own before
            if src in self._our_ips:
                return 1
            return -1
        return 0

    def _detect_local_info(self):
        """Detect local IPs for direction detection."""
        self._our_ips = set()
        self._local_ipv6 = None
        import subprocess
        try:
            result = subprocess.run(["ifconfig", self.interface], capture_output=True, text=True)
            for line in result.stdout.split("\n"):
                line = line.strip()
                if line.startswith("inet ") and "127.0.0.1" not in line:
                    ip = line.split()[1]
                    self._our_ips.add(ip)
                    print(f"    Local IPv4: {ip}")
                if line.startswith("inet6") and "fe80" not in line and "::1" not in line:
                    ip = line.split()[1].split("%")[0]
                    self._our_ips.add(ip)
                    self._local_ipv6 = ip
                    print(f"    Local IPv6: {ip}")
        except Exception:
            pass
        # Fallback
        self._our_ips.add("192.168.1.69")

    def _packet_callback(self, pkt):
        # Accept both IPv4 and IPv6
        has_ip4 = pkt.haslayer(IP)
        has_ip6 = pkt.haslayer(IPv6)
        
        if not has_ip4 and not has_ip6:
            return

        now = time.time()
        size = len(pkt)
        
        # Auto-detect our IPs from outgoing packets (first few packets)
        if len(self.raw_packets) < 50:
            if has_ip4:
                src = pkt[IP].src
                if src.startswith("192.168.") or src.startswith("10."):
                    self._our_ips.add(src)
            if has_ip6:
                src = pkt[IPv6].src
                if src.startswith("2400:") or src.startswith("2600:") or src.startswith("2001:"):
                    # Likely our global IPv6
                    self._our_ips.add(src)
                    if not self._local_ipv6:
                        self._local_ipv6 = src

        direction = self._get_direction(pkt)
        
        # Determine protocol
        if pkt.haslayer(TCP):
            proto = "TCP"
        elif pkt.haslayer(UDP):
            # QUIC runs over UDP on port 443
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
            if sport == 443 or dport == 443:
                proto = "QUIC"
            else:
                proto = "UDP"
        else:
            proto = "Other"

        info = {
            "timestamp": now,
            "size": size,
            "signed_size": size * direction,
            "direction": direction,
            "protocol": proto,
            "payload_size": len(pkt[Raw].load) if pkt.haslayer(Raw) else 0,
            "ttl": pkt[IP].ttl if has_ip4 else pkt[IPv6].hlim if has_ip6 else 0,
        }
        if pkt.haslayer(TCP):
            info["dst_port"] = pkt[TCP].dport
            info["src_port"] = pkt[TCP].sport
            info["window_size"] = pkt[TCP].window
        elif pkt.haslayer(UDP):
            info["dst_port"] = pkt[UDP].dport
            info["src_port"] = pkt[UDP].sport
        self.raw_packets.append(info)
        total = len(self.raw_packets)
        if total % 100 == 0:
            print(f"\r  Captured {total} packets ({len(self._our_ips)} local IPs detected)...", end="", flush=True)

    def _compute_features(self, pkts):
        sizes = [p["size"] for p in pkts]
        signed_sizes = [p["signed_size"] for p in pkts]
        directions = [p["direction"] for p in pkts]
        timestamps = [p["timestamp"] for p in pkts]
        payload_sizes = [p["payload_size"] for p in pkts]
        iats = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
        up_sizes = [s for s, d in zip(sizes, directions) if d == 1]
        down_sizes = [s for s, d in zip(sizes, directions) if d == -1]
        up_iats, down_iats = [], []
        last_up, last_down = None, None
        for ts, d in zip(timestamps, directions):
            if d == 1:
                if last_up is not None: up_iats.append(ts - last_up)
                last_up = ts
            else:
                if last_down is not None: down_iats.append(ts - last_down)
                last_down = ts
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.001
        bursts = []
        cb = 1
        for iat in iats:
            if iat < 0.05: cb += 1
            else:
                if cb > 1: bursts.append(cb)
                cb = 1
        if cb > 1: bursts.append(cb)
        dst_ports = set(p.get("dst_port", 0) for p in pkts)
        sb = [0]*5
        for s in sizes:
            if s < 100: sb[0] += 1
            elif s < 300: sb[1] += 1
            elif s < 600: sb[2] += 1
            elif s < 1000: sb[3] += 1
            else: sb[4] += 1
        tp = max(len(sizes), 1)
        sb = [b/tp for b in sb]

        features = {
            "window_duration": duration,
            "total_packets": len(pkts),
            "total_bytes": sum(sizes),
            "packets_per_sec": len(pkts) / max(duration, 0.001),
            "bytes_per_sec": sum(sizes) / max(duration, 0.001),
            "pkt_size_mean": np.mean(sizes),
            "pkt_size_std": np.std(sizes),
            "pkt_size_min": np.min(sizes),
            "pkt_size_max": np.max(sizes),
            "pkt_size_median": np.median(sizes),
            "pkt_size_q25": np.percentile(sizes, 25),
            "pkt_size_q75": np.percentile(sizes, 75),
            "pkt_size_skew": float(pd.Series(sizes).skew()) if len(sizes) > 2 else 0,
            "pkt_size_iqr": np.percentile(sizes, 75) - np.percentile(sizes, 25),
            "size_bin_tiny": sb[0], "size_bin_small": sb[1], "size_bin_medium": sb[2],
            "size_bin_large": sb[3], "size_bin_jumbo": sb[4],
            "payload_mean": np.mean(payload_sizes),
            "payload_std": np.std(payload_sizes),
            "payload_max": np.max(payload_sizes),
            "payload_zero_ratio": sum(1 for p in payload_sizes if p == 0) / len(payload_sizes),
            "payload_total": sum(payload_sizes),
            "iat_mean": np.mean(iats) if iats else 0,
            "iat_std": np.std(iats) if iats else 0,
            "iat_min": np.min(iats) if iats else 0,
            "iat_max": np.max(iats) if iats else 0,
            "iat_median": np.median(iats) if iats else 0,
            "iat_q25": np.percentile(iats, 25) if iats else 0,
            "iat_q75": np.percentile(iats, 75) if iats else 0,
            "upload_packets": len(up_sizes),
            "download_packets": len(down_sizes),
            "upload_ratio": len(up_sizes) / max(len(pkts), 1),
            "download_ratio": len(down_sizes) / max(len(pkts), 1),
            "up_down_ratio": len(up_sizes) / max(len(down_sizes), 1),
            "up_bytes_total": sum(up_sizes) if up_sizes else 0,
            "up_size_mean": np.mean(up_sizes) if up_sizes else 0,
            "up_size_std": np.std(up_sizes) if up_sizes else 0,
            "up_size_max": np.max(up_sizes) if up_sizes else 0,
            "down_bytes_total": sum(down_sizes) if down_sizes else 0,
            "down_size_mean": np.mean(down_sizes) if down_sizes else 0,
            "down_size_std": np.std(down_sizes) if down_sizes else 0,
            "down_size_max": np.max(down_sizes) if down_sizes else 0,
            "up_iat_mean": np.mean(up_iats) if up_iats else 0,
            "up_iat_std": np.std(up_iats) if up_iats else 0,
            "down_iat_mean": np.mean(down_iats) if down_iats else 0,
            "down_iat_std": np.std(down_iats) if down_iats else 0,
            "n_bursts": len(bursts),
            "avg_burst_size": np.mean(bursts) if bursts else 0,
            "max_burst_size": max(bursts) if bursts else 0,
            "burst_ratio": sum(bursts) / max(len(pkts), 1) if bursts else 0,
            "n_unique_dst_ports": len(dst_ports),
            "tcp_ratio": sum(1 for p in pkts if p["protocol"] == "TCP") / len(pkts),
            "udp_ratio": sum(1 for p in pkts if p["protocol"] == "UDP") / len(pkts),
            "quic_ratio": sum(1 for p in pkts if p["protocol"] == "QUIC") / len(pkts),
            **{f"pss_{i}": signed_sizes[i] if i < len(signed_sizes) else 0 for i in range(PSS_LENGTH)},
            "app": self.app_name,
        }
        return features

    def extract_windows(self):
        if not self.raw_packets:
            return []
        print(f"\n  Extracting sliding windows ({WINDOW_SIZE}s window, {WINDOW_STEP}s step)...")
        self.raw_packets.sort(key=lambda x: x["timestamp"])
        start_t = self.raw_packets[0]["timestamp"]
        end_t = self.raw_packets[-1]["timestamp"]
        features_list = []
        ws = start_t
        valid = 0
        skipped = 0
        while ws + WINDOW_SIZE <= end_t:
            we = ws + WINDOW_SIZE
            window_pkts = [p for p in self.raw_packets if ws <= p["timestamp"] < we]
            if len(window_pkts) >= MIN_PACKETS_PER_WINDOW:
                features_list.append(self._compute_features(window_pkts))
                valid += 1
            else:
                skipped += 1
            ws += WINDOW_STEP
        print(f"  Duration: {end_t - start_t:.1f}s | Valid windows: {valid} | Skipped: {skipped}")
        return features_list

    def start_capture(self, duration=1200):
        print(f"\n{'='*60}")
        print(f"  Capturing: {self.app_name.upper()} | {duration}s | {self.interface}")
        print(f"{'='*60}")
        print(f"\n  Open {self.app_name.upper()} NOW and use it actively!")
        print(f"  Close all other apps. Ctrl+C to stop early.\n")
        try:
            sniff(iface=self.interface, prn=self._packet_callback, store=False, timeout=duration)
        except KeyboardInterrupt:
            pass
        print(f"\n\n  Done! {len(self.raw_packets)} packets captured.")

    def save(self):
        features = self.extract_windows()
        if not features:
            print("  No valid windows. Capture longer (5+ min) or use app more actively.")
            return None
        df = pd.DataFrame(features)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(OUTPUT_DIR, f"{self.app_name}_{ts}.csv")
        df.to_csv(fp, index=False)
        print(f"\n  Saved {len(df)} samples to {fp}")
        return fp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str)
    parser.add_argument("--interface", type=str, default="en0")
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--window", type=float, default=3.0)
    parser.add_argument("--step", type=float, default=1.5)
    args = parser.parse_args()

    global WINDOW_SIZE, WINDOW_STEP
    WINDOW_SIZE = args.window
    WINDOW_STEP = args.step

    print("\n  NetSentinel — App Traffic Collector v2 (Sliding Window)\n")

    if args.interactive or not args.app:
        apps = ["youtube","netflix","spotify","Meta","zoom"]
        print("  Suggested apps:")
        for i, a in enumerate(apps, 1):
            print(f"    {i:2d}. {a}")

        existing = [f.split("_")[0] for f in os.listdir(OUTPUT_DIR) if f.endswith(".csv")]
        if existing:
            print(f"\n  Already captured:")
            for a in sorted(set(existing)):
                print(f"    {a} ({existing.count(a)} sessions)")

        while True:
            app = input("\n  App name (or 'quit'): ").strip().lower().replace(" ", "_")
            if app in ("quit", "q", "exit"): break
            if not app: continue
            dur = input(f"  Duration [{args.duration}s]: ").strip()
            dur = int(dur) if dur else args.duration
            c = WindowCollector(app, args.interface)
            c.start_capture(dur)
            c.save()
            if input("\n  Another app? (y/n): ").strip().lower() != "y": break
    else:
        c = WindowCollector(args.app, args.interface)
        c.start_capture(args.duration)
        c.save()

    print(f"\n  Summary:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
            print(f"    {f}: {len(df)} samples")


if __name__ == "__main__":
    main()
