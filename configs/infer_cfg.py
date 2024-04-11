from utils.common import office_hostname
import socket 


is_office = socket.gethostname() == office_hostname

default_cfg = {
    'batch_size': 8 if not is_office else 2,
    'is_office': is_office,
}