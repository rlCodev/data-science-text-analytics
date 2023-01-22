# Elastic search remote documentation

## Starting elastic search on upunto server
1. Starting elastic search on server:
    `sudo systemctl start elasticsearch`
2. Configure auto start on server boot:
    `sudo systemctl enable elasticsearch`

## Modify ubuntu standard firewall
1. Opening elastic search port: `sudo ufw allow from <trusted remote host> to any port 9200`
2. Enable new rules `sudo ufw enable?`
3. Check firewall status `sudo ufw status`

## Helpful tutorial on setting up kibana or elasticsearch
https://www.digitalocean.com/community/tutorials/how-to-install-elasticsearch-logstash-and-kibana-elastic-stack-on-ubuntu-22-04


## Ubuntu server maintenance
### Server domain: app.leon-remke.jakob-hennighausen.melkonyan-davit.de

1. Connect via ssh
ssh -i <pk-path> <user>@<ip>
2. Edit settings
   - Add user to nginx basic auth:
    `echo "elasticuser:`openssl passwd -apr1`" | sudo tee -a /etc/nginx/htpasswd.users`
   - Edit nginx settings for routing
    `sudo nano /etc/nginx/sites-available/app.leon-remke.jakob-hennighausen.melkonyan-davit.de`
   - Stop/start elasticsearch or kibana
    `$ sudo systemctl <restart | start | stop> <elasticsearch | kibana>`
   - Get service status info 
    `$ sudo systemctl status <elasticsearch | kibana>`