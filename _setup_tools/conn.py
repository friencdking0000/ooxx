import sys
sys.stdout.reconfigure(encoding="utf-8")
from pyorbbecsdk import Pipeline
print("連線：", Pipeline().get_device().get_device_info().get_connection_type())
