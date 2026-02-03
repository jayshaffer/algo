@echo off
REM Start cron service in WSL on Windows boot
REM Add this to Task Scheduler to run at startup

wsl -u root service cron start
