# Prepare
``` bash
sudo apt install python3-tk libxcb-xinerama0
pip3 install -r requirements.txt
```
# Usage
``` bash
# train(put config.json in this dir)
python3 main.py --mode train
# predict
python3 main.py --mode predict
```


## Config file example
put `config.json` file in this directory
### Use EXCEL file as database
- `method`: database type
- `file`: excel file location
- `target`: target col in database
``` json
{
    "method": "excel",
    "file": "./data.xlsx",
    "target": "profile"
}
```

### use MYSQL databse
- `method`: database type
- `host`: mysql ip, use '127.0.0.1' for localhost mysql
- `port`: mysql server port, default is `3306`
- `user`: mysql user name
- `password`: mysql user's password
- `database`: mysql database
- `table`: mysql database's table
- `target`: target col in table
``` json
{
    "method": "mysql",
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123",
    "database": "demo",
    "table": "d2",
    "target": "C1"
}
```