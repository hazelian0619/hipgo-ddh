#!/usr/bin/expect -f

set timeout 300
set host "10.120.18.240"
set user "xlian289-tNxksKkC"
set port "6988"
set password "F5bLCw6Ka1"

spawn ssh -p $port $user@$host
expect {
    "yes/no" { send "yes\r"; exp_continue }
    "password:" { send "$password\r" }
}

# 等待登录完成
expect "$ "

# 进入项目目录
send "cd /home/xlian289-tNxksKkC/hip_project\r"
expect "$ "

# 创建虚拟环境
send "python -m venv venv\r"
expect "$ "

# 激活虚拟环境
send "source venv/bin/activate\r"
expect "$ "

# 使用清华源安装依赖
send "pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple\r"
expect "$ "

# 检查环境
send "python -c \"import torch; print(torch.__version__)\"\r"
expect "$ "

# 保持会话
interact 