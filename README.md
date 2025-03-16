#To Check realtime logs
```
journalctl -u my_script.service -f
```

#To stop the service
```
sudo systemctl stop my_script.services
```

#To disable to to start from boot
```
sudo systemctl disable my_script.services
```

#To re-anble autostart
```
sudo systemctl enable my_script.services
```

script file location
```
sudo nano /etc/systemd/system/my_script.service
```

if you have changes in the script, execute these commands after
```
sudo systemctl daemon-reload
sudo systemctl restart my_script.service
sudo systemctl status my_script.service
```
