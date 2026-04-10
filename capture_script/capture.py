import subprocess
import os

PCAP_FILE = "traffic.pcap"
CAPTURE_SECONDS = 30 
INTERFACE = "enp2s0"

if os.path.exists(PCAP_FILE):
    os.remove(PCAP_FILE)

print(f"Capturing traffic on {INTERFACE} for {CAPTURE_SECONDS}s...")


cmd = [
    "sudo", "tcpdump",
    "-i", INTERFACE,
    "-w", PCAP_FILE,
    "-G", str(CAPTURE_SECONDS),
    "-W", "1"
]


subprocess.run(cmd)

print("Saved. Tcpdump closed automatically.")