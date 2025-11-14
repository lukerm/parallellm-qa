#!/bin/bash

# Capture runtime environment variables and make them available to cron
printenv | sed 's/^\(.*\)$/export \1/g' > /root/docker_env.sh
chmod +x /root/docker_env.sh

# Start cron in foreground
exec cron -f
