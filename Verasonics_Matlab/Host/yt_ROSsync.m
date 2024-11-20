global PA_IMG_pub PA_IMG_msg 

IMGstr = reshape(p_data_trans,1,[]);
PA_IMG_msg.Data = IMGstr;
send(PA_IMG_pub,PA_IMG_msg)

