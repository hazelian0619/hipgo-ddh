#!/usr/bin/expect -f

set timeout 30
set host "10.120.18.240"
set user "xlian289-tNxksKkC"
set port "6988"
set password "F5bLCw6Ka1"

spawn ssh -p $port $user@$host
expect {
    "yes/no" { send "yes\r"; exp_continue }
    "password:" { send "$password\r" }
}
interact 