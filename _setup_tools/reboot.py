import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from pyorbbecsdk import Context
dl = Context().query_devices()
print("devices", dl.get_count())
if dl.get_count():
    dl.get_device_by_index(0).reboot(); print("reboot sent")
