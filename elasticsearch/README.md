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

## Configure Kibana
https://www.digitalocean.com/community/tutorials/how-to-install-elasticsearch-logstash-and-kibana-elastic-stack-on-ubuntu-22-04


## Kibana openssl admin user:
kibanaadmin:$apr1$1Ve/kfXB$3ai5DrR1ebeqjG45wM3k0.

