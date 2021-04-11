# Deploy OpenBox Service

This tutorial helps you deploy an **OpenBox** service. If you are an **OpenBox** service user, please refer to 
the [Service User Tutorial](./use_service).


## 1 Install OpenBox from Source

Installation Requirements:

+ Python >= 3.5

+ SWIG == 3.0

Make sure to install SWIG correctly before you install OpenBox.

To install SWIG, please refer to [SWIG Installation Guide](../installation/install_swig.md)

Then, clone the source code from Github to the server you want to deploy OpenBox service.
Please use the following command on the command line:

```bash
git clone https://github.com/thomas-young-2013/lite-bo.git
cd lite-bo
python setup.py install
```


## 2 Initialize MongoDB

**OpenBox** relies [MongoDB](https://www.mongodb.com) to store user and task information.

### 2.1 Install and Run MongoDB

Please install and run MongoDB before running **OpenBox** service. 
For MongoDB installation guides, refer to the following links:

+ <https://docs.mongodb.com/guides/server/install/>

+ <https://docs.mongodb.com/manual/installation/>

You need to create a MongoDB user and set **auth=true** when starting MongoDB.
Please record the **IP** and **port** of your database.

### 2.2 Modify service.conf File

After starting MongoDB, modify <font color=#FF0000>**"lite-bo/conf/service.conf"**</font> to set database information.
If this is your first time running the service, create the <font color=#FF0000>**service.conf**</font> file by copying 
the template config file from <font color=#FF0000>**"lite-bo/conf/template/service.conf.template"**</font> to 
<font color=#FF0000>**"lite-bo/conf/"**</font> and rename it to <font color=#FF0000>**service.conf**</font>.

The contents of **service.conf** are as follows:

```
[database]
database_address=127.0.0.1
database_port=27017
user=xxxx
password=xxxx
```

Please set database IP, port, user and password accordingly.

**Caution:** We have added **service.conf** to **.gitignore**. Do not push this file to **Github** to
prevent the disclosure of private information.


## 3 Set up Email Registration Service

### 3.1 Prepare an Email for Registration Service

**OpenBox** requires users to register accounts by email. Please prepare an email to send activation link to users.
Enable SMTP authentication of the email and you should get a secret key from email provider.

### 3.2 Modify litebo/artifact/artifact/settings.py

Then, modify <font color=#FF0000>**"litebo/artifact/artifact/settings.py"**</font> to set email information for
registration service. Please fill in the following lines:

```python
EMAIL_HOST = 'smtp.xxxx.com'
EMAIL_PORT = 465
EMAIL_HOST_USER = 'xxxx@xxxx.com'
EMAIL_HOST_PASSWORD = 'xxxx'
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER
```

+ **EMAIL_HOST:** SMTP host of email service provider. 'smtp.gmail.com' for example.
+ **EMAIL_PORT:** SMTP port of email. Get the port from email service provider. You may try port 25, 587 
or other ports if port 465 doesn't work.
+ **EMAIL_HOST_USER:** Your email address for registration service.
+ **EMAIL_HOST_PASSWORD:** Your secret key for SMTP authentication of your email.

**Caution:** Do not push the file with private information to **Github** to prevent leakage.

## 4 Start/Stop OpenBox Service

Finally, after setting up database and registration service email, you can start up **OpenBox** service.

To **start the service**, run the <font color=#FF0000>**manage_service.sh script**</font> by the following command:

```bash
cd <path to the source code>/lite-bo
./scripts/manage_service.sh start
```

The script will run **OpenBox** service in the background. Default service port is 11425.
You can modify the script to change service port.

Then, visit <http://127.0.0.1:11425/user_board/index/> (replace "127.0.0.1:11425" by your server ip:port)
to see whether your service started successfully.
You can also try to create an account and run a task to test your **OpenBox** service. 
For more detailed guidance, please refer to the [Service User Tutorial](./use_service).

To **stop the service**, run the <font color=#FF0000>**manage_service.sh script**</font> by the following command:

```bash
cd <path to the source code>/lite-bo
./scripts/manage_service.sh stop
```

