
import os
try:
    with open('backend_debug.log', 'r') as f:
        lines = f.readlines()
        print(''.join(lines[-200:]))
except Exception as e:
    print(f"Error reading log: {e}")
