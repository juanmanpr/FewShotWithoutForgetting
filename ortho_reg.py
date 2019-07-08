import torch

def l2_reg_ortho(weight_matrix, device):
    
    W = weight_matrix
    
    cols = W[0].numel()
    rows = W.shape[0]
    w1 = W.view(-1,cols)
    wt = torch.transpose(w1,0,1)
    if (rows > cols):
        m  = torch.matmul(wt,w1)
        ident = torch.eye(cols,cols,requires_grad=True).to(device)
    else:
        m = torch.matmul(w1,wt)
        ident = torch.eye(rows,rows,requires_grad=True).to(device)

    w_tmp = (m - ident)
    b_k = torch.rand(w_tmp.shape[1],1).to(device)

    v1 = torch.matmul(w_tmp, b_k)
    norm1 = torch.norm(v1,2)
    v2 = torch.div(v1,norm1)
    v3 = torch.matmul(w_tmp,v2)

    l2_reg = (torch.norm(v3,2))**2
                    
    return l2_reg
