#!/bin/bash

cd $HOME/parallellm-qa

PYTHONPATH=. LOG_LEVEL=INFO python -m src.run_login
PYTHONPATH=. LOG_LEVEL=info python -m src.run_chats