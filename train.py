# trainToyOTflow.py
# training driver for the two-dimensional toy problems
import argparse
import sys
import os
import time
import datetime
import torch.optim as optim
import numpy as np
import math
import torch
import numpy as np
from torch.nn.functional import pad
from src.Phi import *
import lib.toy_data as toy_data
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('OT-Flow')
parser.add_argument(
    '--data', choices=['square', 'porous', 'swissroll', '8gaussians','2gaussians', '1gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='8gaussians'
)

parser.add_argument("--nt"    , type=int, default=1, help="number of time steps")
parser.add_argument("--nt_val", type=int, default=1, help="number of time steps for validation")
parser.add_argument("--tau", type=float, default=0.001, help="the time stepsize for the outer JKO iteration")
parser.add_argument('--alph'  , type=str, default='1.0,1.0,0.0')
parser.add_argument('--m'     , type=int, default=128)
parser.add_argument('--nTh'   , type=int, default=3)

parser.add_argument('--niters'        , type=int  , default=4000)
parser.add_argument('--batch_size'    , type=int  , default=200)
parser.add_argument('--val_batch_size', type=int  , default=200)

parser.add_argument('--lr'          , type=float, default=0.0001)
parser.add_argument("--drop_freq"   , type=int  , default=500, help="how often to decrease learning rate")
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--lr_drop'     , type=float, default=1.00001)
parser.add_argument('--optim'       , type=str  , default='adam', choices=['adam'])
parser.add_argument('--prec'        , type=str  , default='single', choices=['single','double'], help="single or double precision")

parser.add_argument('--save'    , type=str, default='experiments')
parser.add_argument('--viz_freq', type=int, default=200)
parser.add_argument('--val_freq', type=int, default=1)
parser.add_argument('--gpu'     , type=int, default=0)
parser.add_argument('--sample_freq', type=int, default=25)

parser.add_argument('--n_tau', type=int, default=0)

args = parser.parse_args()

args.alph = [float(item) for item in args.alph.split(',')]

# get precision type
if args.prec =='double':
    prec = torch.float64
else:
    prec = torch.float32

# get timestamp for saving models
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

os.makedirs(args.save, exist_ok=True)
os.makedirs(f"{args.save}/figs", exist_ok=True)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def compute_loss_gf(net, x, nt, tau, n_tau, net_list, rho, z):
    Jc , cs = OTFlowProblemGradientFlowsPorous(x, rho, net, [0,1], nt=nt, tau=tau, n_tau=n_tau, net_list=net_list, stepper="rk4", alph=net.alph, z=z)
    return Jc, cs

def plot_scatter(x_next, args, n_tau, sPath):
    fig, ax = plt.subplots(1,1,figsize=(4,4))
    x_next = x_next.numpy()
    ax.scatter(x_next[:,0], x_next[:,1], marker='.')
    ax.set_xlim([-4.1,4.1])
    ax.set_ylim([-4.1,4.1])
    plt.tight_layout()
    ax.set_aspect(True)
    plt.savefig(sPath)
    plt.close('all')

def plot_scatter_color(x_next, args, n_tau, color, sPath):
    t0  = 0.001
    tau = args.tau
    t = tau * (n_tau+1)
    m_power = 2
    A = (4 * np.pi * m_power * (t+t0)) ** ((1-m_power)/m_power)
    B = (m_power-1)/(4*m_power**2*(t+t0))
    r = np.sqrt(A/B)
    circle1 = plt.Circle((0, 0), r, color='g', fill=False)
    
    fig, ax = plt.subplots(1,1,figsize=(4,4))

    x_next = x_next.numpy()
    ax.scatter(x_next[:,0], x_next[:,1], c=color, marker='.')
    ax.set_xlim([-1.4,1.4])
    ax.set_ylim([-1.4,1.4])
    
    ax.add_patch(circle1)
    
    plt.tight_layout()
    ax.set_aspect(True)
    plt.savefig(sPath)
    plt.close('all')


def multiple_exp(Tx):
    pos1= torch.tensor([[2.0,-2.0]])
    pos2= torch.tensor([[-2.0,2.0]])
    pos3= torch.tensor([[2.0,2.0]])
    pos4= torch.tensor([[-2.0,-2.0]])
    return 1/(2*np.pi) * ( torch.exp(-torch.sum((Tx-pos1)**2, dim=1)/(2 * 0.5**2)) 
                         + torch.exp(-torch.sum((Tx-pos2)**2, dim=1)/(2 * 0.5**2))  
                         + torch.exp(-torch.sum((Tx-pos3)**2, dim=1)/(2 * 0.5**2))
                         + torch.exp(-torch.sum((Tx-pos4)**2, dim=1)/(2 * 0.5**2)) )

def compute_U(rho_next, Tx=None):
    if Tx != None:
        return rho_next * ( torch.log(rho_next) - torch.log(multiple_exp(Tx) + 1e-5) - 1 )
    else:
        return rho_next * ( torch.log(rho_next) - 1 )

def OTFlowProblemGradientFlowsPorous(x, rho, Phi, tspan , nt, tau, n_tau, net_list, stepper="rk4", alph =[1.0,1.0,1.0], z=None):
    """

    Evaluate objective function of OT Flow problem; see Eq. (8) in the paper.
    
    :param x:       input data tensor nex-by-d
    :param rho:     input rho nex-by-1
    :param Phi:     neural network
    :param tspan:   time range to integrate over, ex. [0.0 , 1.0]
    :param nt:      number of time steps
    :param n_tau: nth step in gradient flows
    :param net_list: list of nets of length n_tau
    :param stepper: string "rk1" or "rk4" Runge-Kutta schemes
    :param alph:    list of length 3, the alpha value multipliers
    :return:
        Jc - float, objective function value dot(alph,cs)
        cs - list length 5, the five computed costs
    """
    h = (tspan[1]-tspan[0]) / nt

    d = z.shape[1]-3 # dimension for x

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    rho_next = rho
    tk = tspan[0]

    # given the data from the list of nets, compute z for this iteration
    if stepper=='rk4':
        for k in range(nt):
            z = stepRK4(odefun, z, Phi, alph, tk, tk + h)
            tk += h
    elif stepper=='rk1':
        for k in range(nt):
            z = stepRK1(odefun, z, Phi, alph, tk, tk + h)
            tk += h

    # interaction cost
    n  = z.shape[0]
    Tx = z[:,0:d]

    rho_next = rho_next / torch.exp(z[:,d]) + 1e-5
    terminal_cost = (compute_U(rho_next,Tx) / (rho_next)).mean()
    costL  = torch.mean(z[:,-2]) * 0.5
    costC  = terminal_cost * tau
    costR = 0

    cs = [costL, costC, costR]
    return sum(i[0] * i[1] for i in zip(cs, alph)) , cs

def stepRK4(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 4 integration scheme
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the OT-Flow Problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """

    h = t1 - t0 # step size
    z0 = z

    K = h * odefun(z0, t0, Phi, alph=alph)
    z = z0 + (1.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + 0.5*K , t0+(h/2) , Phi, alph=alph)
    z += (2.0/6.0) * K

    K = h * odefun( z0 + K , t0+h , Phi, alph=alph)
    z += (1.0/6.0) * K

    return z


def get_samples_next_time_step_including_det(x, rho, net, net_list, nt, n_tau, stepper="rk4", alph =[1.0,1.0,1.0] ):
    h = 1.0 / nt

    # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
    z = pad(x, (0, 3, 0, 0), value=0)

    d = z.shape[1]-3 # dimension for x

    # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
    rho_next = rho
    with torch.no_grad(): 
        for n in range(n_tau):
            tk = 0
            for k in range(nt):
                z = stepRK4(odefun, z, net_list[n], alph, tk, tk + h)
                tk += h
            rho_next = rho_next / torch.exp(z[:,d]) + 1e-6
            z = pad(z[:,0:d], (0,3,0,0), value=0)
        tk = 0
        for k in range(nt):
            z = stepRK4(odefun, z, net, alph, tk, tk + h)
            tk += h
        rho_next = rho_next / torch.exp(z[:,d]) + 1e-6
    return z[:,:d], rho_next

def stepRK1(odefun, z, Phi, alph, t0, t1):
    """
        Runge-Kutta 1 / Forward Euler integration scheme.  Added for comparison, but we recommend stepRK4.
    :param odefun: function to apply at every time step
    :param z:      tensor nex-by-d+4, inputs
    :param Phi:    Module, the Phi potential function
    :param alph:   list, the 3 alpha values for the mean field game problem
    :param t0:     float, starting time
    :param t1:     float, end time
    :return: tensor nex-by-d+4, features at time t1
    """
    z += (t1 - t0) * odefun(z, t0, Phi, alph=alph)
    return z

def odefun(x, t, net, alph=[1.0,1.0,1.0]):
    """
    neural ODE combining the characteristics and log-determinant (see Eq. (2)), the transport costs (see Eq. (5)), and
    the HJB regularizer (see Eq. (7)).

    d_t  [x ; l ; v ; r] = odefun( [x ; l ; v ; r] , t )

    x - particle position
    l - log determinant
    v - accumulated transport costs (Lagrangian)
    r - accumulates violation of HJB condition along trajectory
    """
    n, d_extra = x.shape
    d = d_extra - 3

    z = pad(x[:, :d], (0, 1, 0, 0), value=t) # concatenate with the time t

    # print("zshape:", z.shape, "d", d)

    gradPhi, trH = net.trHess(z)

    dx = -(1.0/alph[0]) * gradPhi[:,0:d]
    dl = -(1.0/alph[0]) * trH.unsqueeze(1)
    dv = 0.5 * torch.sum(torch.pow(dx, 2) , 1 ,keepdims=True)
    dr = torch.abs(  -gradPhi[:,-1].unsqueeze(1) + alph[0] * dv  ) 
    
    return torch.cat( (dx,dl,dv,dr) , 1  )

if __name__ == '__main__':

    torch.set_default_dtype(prec)
    cvt = lambda x: x.type(prec).to(device, non_blocking=True)

    # neural network for the potential function Phi
    d      = 2
    alph   = args.alph
    nt     = args.nt
    nt_val = args.nt_val
    tau    = args.tau
    nTh    = args.nTh
    m      = args.m
    net = Phi(nTh=nTh, m=args.m, d=d, alph=alph)
    net = net.to(prec).to(device)
    n_tau = args.n_tau


    b1 = 0.5
    b2 = 0.999
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(b1, b2))
    net_list = []

    for i in range(n_tau):
        filename = os.path.join(args.save, '{:}_alph{:}_{:}_m{:}_n{}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m,i))
        print(f"loading {filename}")
        checkpt  = torch.load(filename, map_location=lambda storage, loc: storage)
        net1     = Phi(nTh=nTh, m=m, d=d, alph=alph)
        prec1    = checkpt['state_dict']['A'].dtype
        net1     = net1.to(prec1).to(device)
        net1.load_state_dict(checkpt['state_dict'])
        net_list.append(net1)

        if i == n_tau - 1:
            net.load_state_dict(checkpt['state_dict'])
            x0 = checkpt['x0']
            rho = checkpt['rho']
            energy_list = checkpt['energy']
            energy_list.append(0)
            error_list = checkpt['error']
            error_list.append(0)


    # if it is the first outer iteration then sample random points
    if n_tau == 0:
        x0,rho = toy_data.inf_train_gen(args.data, batch_size=args.batch_size, dim=d)
        print(x0.shape, rho.shape)
        x0 = cvt(torch.from_numpy(x0))
        rho = cvt(torch.from_numpy(rho))
        energy_list = [0]
        error_list = [0]

        # for plotting rho(t=0)
        x0plot,rhoplot = toy_data.inf_train_gen(args.data, batch_size=1000, dim=d)
        x0plot = cvt(torch.from_numpy(x0plot))
        rhoplot = cvt(torch.from_numpy(rhoplot))
        plot_scatter_color(x0plot, args, n_tau-1, rhoplot, sPath = os.path.join(args.save, 'figs', f"rho_{n_tau}.png"))
        del x0plot
        del rhoplot
    # else choose the same points from the previous outer iteration which is happening in previous lines.

    # print(net)
    print("-------------------------")
    print("DIMENSION={:}  m={:}  nTh={:}   alpha={:}".format(d,m,nTh,alph))
    print("nt={:}   nt_val={:}   tau={:}  n_tau={:}".format(nt,nt_val,tau,n_tau))
    print("-------------------------")
    print(str(optim)) # optimizer info
    print("data={:} batch_size={:} gpu={:}".format(args.data, args.batch_size, args.gpu))
    print("maxIters={:} val_freq={:} viz_freq={:}".format(args.niters, args.val_freq, args.viz_freq))
    print("saveLocation = {:}".format(args.save))
    print("-------------------------\n")

    end = time.time()
    best_loss = float('inf')
    bestParams = None

    # setup data [nSamples, d]
    # use one batch as the entire data set


    x0val,rhoval = toy_data.inf_train_gen(args.data, batch_size=args.val_batch_size, dim=d)
    x0val = cvt(torch.from_numpy(x0val))

    log_msg = (
        '{:5s}  {:6s}   {:9s}  {:9s}  {:9s}  {:9s}  {:9s}      {:9s}  {:9s}  {:9s}  {:9s}  '.format(
            'iter', ' time','loss', 'L (L_2)', 'C (loss)', 'R (HJB)', 'rel loss', 'valLoss', 'valL', 'valC', 'valR'
        )
    )
    print(log_msg)

    net.train()

    previous_loss = 1

    ccc = True # for plotting in between

    global_loss = 1

    best_params = net.state_dict()
    time_meter = [time.time()]

    for itr in range(0, args.niters + 1):
        # train sampling
        if itr % 100 == 0:
            x0,rho = toy_data.inf_train_gen(args.data, batch_size=args.batch_size)
            x0 = cvt(torch.from_numpy(x0))
            rho = cvt(torch.from_numpy(rho))

            h = 1.0 / nt

            # initialize "hidden" vector to propogate with all the additional dimensions for all the ODEs
            # z = pad(x0, (0, 3, 0, 0), value=0)
            z = torch.zeros((x0.shape[0], d+3)) ; z[:,:d] = x0
            # get the inital density from net_list. Given an initial density at t=0, iterate through n_tau - 1
            rho_next = rho
            with torch.no_grad(): 
                for n in range(n_tau):
                    tk = 0
                    for k in range(nt):
                        z = stepRK4(odefun, z, net_list[n], alph, tk, tk + h)
                        tk += h
                        rho_next = rho_next / torch.exp(z[:,d]) + 1e-5
                    # z = pad(z[:,0:d], (0,3,0,0), value=0)
                    z[:,d:] = 0

        optim.zero_grad()
        loss, costs  = compute_loss_gf(net, x0, nt=nt, tau=tau, n_tau=n_tau, net_list=net_list, rho=rho, z=z)
        loss.backward()
        optim.step()

        time_meter.append(time.time())

        # stopping condition
        tol = 1e-10
        rel_loss = abs(previous_loss - loss)/abs(previous_loss)
        # if rel_loss < tol or itr == args.niters:
        if itr == args.niters:
            # best_loss   = test_loss.item()
            # best_costs = test_costs
            with torch.no_grad():
                best_loss   = loss.item()
                best_costs  = costs
                best_params = net.state_dict()

                x0val,rhoval = toy_data.inf_train_gen(args.data, batch_size=1000, dim=d)
                x0val  = cvt(torch.from_numpy(x0val))
                rhoval = cvt(torch.from_numpy(rhoval))

                # compute samples at the next time step
                x_next, rho_next = get_samples_next_time_step_including_det(x0val, rhoval, net, net_list, nt, n_tau)

                mean = torch.mean(x_next, dim=0)
                var  = torch.var(x_next)

                print(f"mean = {mean} variance = {var}")
                plot_scatter_color(x_next, args, n_tau, rho_next, sPath = os.path.join(args.save, 'figs', f"rho_{n_tau+1}.png"))

                t0  = 0.001
                tau = args.tau
                t = tau * (n_tau+1)

                m_power = 2
                A = (4 * np.pi * m_power * (t+t0)) ** ((1-m_power)/m_power)
                B = (m_power-1)/(4*m_power**2*(t+t0))
                rho_next_exact = np.zeros((x_next.shape[0]))
                for i,x in enumerate(x_next):
                    xnorm2 = torch.sum(x**2)
                    rho_next_exact[i] = max(A - B * xnorm2, 0)**(1/(m_power-1))

                rho_next = rho_next.numpy()
                energy_list[-1] = costs[1]
                error_list[-1]  = np.mean((rho_next - rho_next_exact)**2)
                np.savetxt("error.csv", np.array(error_list), delimiter=",")
                
                do_save = True # tmp var for saving the data
                if do_save == True:
                    torch.save({
                        'args': args,
                        'state_dict': best_params,
                        'x0': x_next,
                        'rho': rho_next,
                        'error': error_list,
                        'energy': energy_list,
                    }, os.path.join(args.save, '{:}_alph{:}_{:}_m{:}_n{}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m,n_tau)))
                    print(f"breaking rel loss={rel_loss}")
                    print(f"error saved in {os.path.join(args.save, '{:}_alph{:}_{:}_m{:}_n{}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m,n_tau))}")
            break

        previous_loss = loss

        log_message = (
            '{:05d}  {:6.3f}   {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}  {:9.3e}'.format(
                itr, time_meter[-1] - time_meter[-2] , loss, costs[0], costs[1], costs[2], rel_loss
            )
        )

        if loss.item() < global_loss:
            global_loss = loss.item()
            best_params = net.state_dict()


        if itr % 1000 == 0:
            net.load_state_dict(best_params)

        # create plots
        if itr % args.viz_freq == 0:
            print(log_message) # print iteration
            with torch.no_grad():
                # compute samples at the next time step
                x_next, rho_next = get_samples_next_time_step_including_det(x0val, rhoval, net, net_list, nt, n_tau)
                mean = torch.mean(x_next, dim=0)
                var  = torch.var(x_next)
                x_next, rho_next = get_samples_next_time_step_including_det(x0val, rhoval, net, net_list, nt, n_tau)
                plot_scatter_color(x_next, args, n_tau, rho_next, sPath = os.path.join(args.save, 'figs', f'n_tau_{n_tau}_itr_{itr}.png'))

        # shrink step size
        if itr % args.drop_freq == 0:
            for p in optim.param_groups:
                p['lr'] /= args.lr_drop

        end = time.time()

    print("Training Time: {:} seconds".format(time_meter[-1] - time_meter[0]))
    print('Training has finished.  ' + '{:}_alph{:}_{:}_m{:}_checkpt.pth'.format(args.data,int(alph[1]),int(alph[2]),m))





