# Print decoded event batches from a Prophesee GenX320 connected to an OpenMV N6.
# Run in OpenMV IDE with the GenX320 connected to the N6 CSI connector.
# Events are printed in the IDE Serial Terminal. Printing every event is slow
# and is intended for a basic test, not high-rate recording.

import csi
from ulab import numpy as np

# Must be a power of two between 1024 and 65536.
EVENT_BUFFER_SIZE = 1024

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(
    csi.IOCTL_GENX320_SET_MODE,
    csi.GENX320_MODE_EVENT,
    EVENT_BUFFER_SIZE,
)

# Each row: [type, seconds, milliseconds, microseconds, x, y]
events = np.zeros((EVENT_BUFFER_SIZE, 6), dtype=np.uint16)

print("Reading GenX320 events. Move something in front of the sensor.")

while True:
    count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

    for i in range(count):
        event = events[i]
        event_type = int(event[0])

        if event_type == csi.PIX_ON_EVENT:
            label = "ON"
        elif event_type == csi.PIX_OFF_EVENT:
            label = "OFF"
        else:
            label = "TRIGGER"

        print(
            label,
            "time={}.{:03d}.{:03d}".format(
                int(event[1]), int(event[2]), int(event[3])
            ),
            "x={}, y={}".format(int(event[4]), int(event[5])),
        )
