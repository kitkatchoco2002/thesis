#To Check realtime logs
```
journalctl -u my_script.services -f
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
