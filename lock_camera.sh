#!/bin/bash
DEV=/dev/video0

# Force manual exposure
v4l2-ctl -d $DEV -c auto_exposure=1

# Give driver a moment
sleep 0.3

# Exposure + gain (your working values)
v4l2-ctl -d $DEV -c exposure_time_absolute=2000
v4l2-ctl -d $DEV -c gain=200

# Disable autofocus (fixed scene)
v4l2-ctl -d $DEV -c focus_automatic_continuous=0

# Keep white balance automatic (works well here)
v4l2-ctl -d $DEV -c white_balance_automatic=1
