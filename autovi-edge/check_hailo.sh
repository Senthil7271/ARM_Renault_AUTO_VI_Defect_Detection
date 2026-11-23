#!/bin/bash
echo "============================================================"
echo " HAILO-8 HARDWARE PRESENCE & LINK STATUS DIAGNOSTIC"
echo "============================================================"

echo -e "\n[1/4] Checking PCIe Bus Devices (lspci):"
HAILO_PCI=$(lspci -nn | grep -i -E "hailo|1e60")
if [ -n "$HAILO_PCI" ]; then
    echo -e "  \033[92m[FOUND]\033[0m $HAILO_PCI"
else
    echo -e "  \033[91m[NOT DETECTED]\033[0m No Hailo device on PCIe bus."
fi

echo -e "\n[2/4] Checking Kernel Device Node (/dev/hailo*):"
if ls /dev/hailo* 1>/dev/null 2>&1; then
    echo -e "  \033[92m[FOUND]\033[0m $(ls -l /dev/hailo*)"
else
    echo -e "  \033[91m[NOT DETECTED]\033[0m /dev/hailo0 does not exist."
fi

echo -e "\n[3/4] Running Official Hailo Scanner (hailortcli scan):"
hailortcli scan 2>&1 | sed 's/^/  /'

echo -e "\n[4/4] Checking BCM2712 PCIe 1x Link Status in dmesg:"
LINK_STATUS=$(dmesg | grep -i "1000110000.pcie" | grep -i "link" | tail -n 1)
if echo "$LINK_STATUS" | grep -qi "link up"; then
    echo -e "  \033[92m[LINK UP]\033[0m $LINK_STATUS"
else
    echo -e "  \033[91m[LINK DOWN]\033[0m $LINK_STATUS"
fi

echo "============================================================"
