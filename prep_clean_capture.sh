#!/bin/bash
# ═══════════════════════════════════════════════════
# NetSentinel — Clean Capture Prep Script (Mac)
# Pauses background network activity for clean data
# ═══════════════════════════════════════════════════
#
# Usage: 
#   sudo bash prep_clean_capture.sh start    (pause background stuff)
#   sudo bash prep_clean_capture.sh stop     (resume everything)

ACTION=${1:-start}

if [ "$ACTION" == "start" ]; then
    echo ""
    echo "══════════════════════════════════════════"
    echo "  🧹 Preparing for Clean App Capture"
    echo "══════════════════════════════════════════"
    echo ""

    # 1. Disable Spotlight indexing (generates network traffic)
    echo "  [1/7] Pausing Spotlight indexing..."
    mdutil -a -i off 2>/dev/null
    
    # 2. Disable iCloud sync
    echo "  [2/7] Note: Manually pause iCloud sync in System Settings"
    
    # 3. Kill common background apps that generate traffic
    echo "  [3/7] Killing background network apps..."
    
    KILL_APPS=(
        "Dropbox"
        "OneDrive"
        "Google Drive"
        "Slack"
        "Discord"
        "Telegram"
        "WhatsApp"
        "Messenger"
        "Microsoft Teams"
        "zoom.us"
        "Spotify"
        "Music"
        "Mail"
        "Calendar"
        "Reminders"
        "Notes"
        "News"
        "Stocks"
        "Weather"
        "Maps"
        "Photos"
        "App Store"
        "Software Update"
        "Creative Cloud"
        "Adobe"
        "Steam"
    )
    
    for app in "${KILL_APPS[@]}"; do
        pkill -f "$app" 2>/dev/null
    done
    
    # 4. Disable automatic software updates
    echo "  [4/7] Pausing software updates..."
    softwareupdate --schedule off 2>/dev/null
    defaults write com.apple.SoftwareUpdate AutomaticDownload -bool false 2>/dev/null
    
    # 5. Disable Time Machine
    echo "  [5/7] Pausing Time Machine..."
    tmutil disable 2>/dev/null
    
    # 6. Flush DNS cache (clears pending DNS lookups)
    echo "  [6/7] Flushing DNS cache..."
    dscacheutil -flushcache 2>/dev/null
    killall -HUP mDNSResponder 2>/dev/null
    
    # 7. Kill browser tabs (reminder)
    echo "  [7/7] Close ALL browser tabs before capture!"
    
    echo ""
    echo "  ✅ System prepared for clean capture!"
    echo ""
    echo "  ══════════════════════════════════════"
    echo "  IMPORTANT — Before capturing:"
    echo "  ══════════════════════════════════════"
    echo "  1. Close ALL browser windows/tabs"
    echo "  2. Quit ALL apps except Terminal"
    echo "  3. Wait 10 seconds for traffic to settle"
    echo "  4. Then run: sudo python collect_app_traffic.py --interactive"
    echo "  5. Open ONLY the target app and use it"
    echo ""
    echo "  To restore everything after:"
    echo "    sudo bash prep_clean_capture.sh stop"
    echo ""

elif [ "$ACTION" == "stop" ]; then
    echo ""
    echo "══════════════════════════════════════════"
    echo "  🔄 Restoring Normal Operation"
    echo "══════════════════════════════════════════"
    echo ""
    
    # Re-enable Spotlight
    echo "  [1/4] Resuming Spotlight..."
    mdutil -a -i on 2>/dev/null
    
    # Re-enable software updates
    echo "  [2/4] Resuming software updates..."
    softwareupdate --schedule on 2>/dev/null
    defaults write com.apple.SoftwareUpdate AutomaticDownload -bool true 2>/dev/null
    
    # Re-enable Time Machine
    echo "  [3/4] Resuming Time Machine..."
    tmutil enable 2>/dev/null
    
    # Flush DNS
    echo "  [4/4] Flushing DNS..."
    dscacheutil -flushcache 2>/dev/null
    
    echo ""
    echo "  ✅ Everything restored! You can reopen your apps now."
    echo ""

else
    echo "Usage: sudo bash prep_clean_capture.sh [start|stop]"
    echo "  start  — Pause background services for clean capture"
    echo "  stop   — Resume everything"
fi
