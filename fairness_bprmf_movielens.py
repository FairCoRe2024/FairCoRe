import argparse
import calendar
import gc
import pickle
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from model import *
from utils import *
import csv
import matplotlib.pyplot as plt


def construct_bystep(e_xu, sens, args):
    x_norm = F.normalize(e_xu).cpu().detach().numpy()
    sens = sens.cpu().numpy()
    size = 1024
    step = x_norm.shape[0] // size + 1
    diff_like, same_unlike = None, None
    dense = x_norm @ x_norm.T
    min_vals = np.min(dense, axis=1)
    max_vals = np.max(dense, axis=1)
    min_arr = np.tile(min_vals, (dense.shape[1], 1)).T
    max_arr = np.tile(max_vals, (dense.shape[1], 1)).T
    pos_all_list = []
    neg_all_list = []
    for i in tqdm(range(step)):
        start, end = i * size, (i + 1) * size
        if end > x_norm.shape[0]:
            end = x_norm.shape[0]

        real_size = end - start

        x_norm_step = x_norm[start:end]

        like_dense = x_norm_step @ x_norm.T
        like_dense = (like_dense - min_arr[start:end, :]) / (max_arr[start:end, :] - min_arr[start:end, :])

        unlike_dense = like_dense.copy()

        like_dense[like_dense >= args.u_threshold] = 1
        like_dense[like_dense < args.u_threshold] = 0

        idx_unlike_zero = (unlike_dense >= args.l_threshold)
        idx_unlike_one = (unlike_dense < args.l_threshold)
        unlike_dense[idx_unlike_zero] = 0
        unlike_dense[idx_unlike_one] = 1

        sens_step = sens[start:end]

        sens_diff = np.not_equal(sens_step[:, None], sens)
        sens_same = np.equal(sens_step[:, None], sens)

        diff_like = sens_diff * like_dense
        same_unlike = sens_same * unlike_dense

        np.fill_diagonal(diff_like[:real_size, start:start + real_size], 1)
        np.fill_diagonal(same_unlike[:real_size, start:start + real_size], 1)

        diff_like = diff_like[:real_size, :]
        same_unlike = same_unlike[:real_size, :]

        for i in range(real_size):
            pos_all_list.append(same_unlike[i].nonzero()[0])
            neg_all_list.append(diff_like[i].nonzero()[0])

    return pos_all_list, neg_all_list


def sample(pos_all_list, neg_all_list, n_users, args):
    pos_indices = np.random.randint(0, pos_all_list[0].size, size=(n_users, args.k))
    neg_indices = np.random.randint(0, neg_all_list[0].size, size=(n_users, args.k))

    pos_select_list = pos_all_list[np.arange(n_users)[:, np.newaxis], pos_indices]
    neg_select_list = neg_all_list[np.arange(n_users)[:, np.newaxis], neg_indices]

    return torch.from_numpy(pos_select_list).long().view(-1), torch.from_numpy(neg_select_list).long().view(-1)


def train_semigcn(gcn, gcn2, sens, n_users, n_items, e_xu, e_xi, args, classes_num, device='cuda:0'):
    optimizer = optim.Adam(gcn.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(gcn2.parameters(), lr=args.lr)
    sens = torch.tensor(sens).to(torch.long)
    sens_opp = vector_transform(sens, classes_num).to(device)
    sens = sens.to(device)

    criterion = nn.MSELoss(reduction='sum')
    pos_all_list, neg_all_list = construct_bystep(e_xu, sens, args)
    user_list = torch.from_numpy(np.repeat(np.arange(n_users), args.k)).long()
    for i in range(n_users):
        if (len(pos_all_list[i]) > 1):
            pos_all_list[i] = np.delete(pos_all_list[i], np.where(pos_all_list[i] == i)[0])
        if (len(neg_all_list[i]) > 1):
            neg_all_list[i] = np.delete(neg_all_list[i], np.where(neg_all_list[i] == i)[0])
    max_pos_len = max(len(pos) for pos in pos_all_list)
    max_neg_len = max(len(neg) for neg in neg_all_list)

    pos_all_list_padded = []
    neg_all_list_padded = []
    for i in range(n_users):
        pos_list_extended = np.random.choice(pos_all_list[i], size=max_pos_len, replace=True)
        pos_all_list_padded.append(pos_list_extended)

        neg_list_extended = np.random.choice(neg_all_list[i], size=max_neg_len, replace=True)
        neg_all_list_padded.append(neg_list_extended)
    pos_all_list = np.array(pos_all_list_padded)
    neg_all_list = np.array(neg_all_list_padded)
    for epoch in tqdm(range(args.sim_epochs)):
        pos_select_list, neg_select_list = sample(pos_all_list, neg_all_list, n_users, args)
        e_su, e_si, su, si = gcn()
        classify_loss = F.cross_entropy(su.squeeze(), sens.squeeze())
        sim_loss = calc_sim_loss(e_su, user_list, pos_select_list, neg_select_list)
        sim_loss = sim_loss * args.s_reg

        loss = classify_loss + sim_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = classify_loss.item()
        sim = sim_loss.item()

    gcn.fc.requires_grad = False
    gcn2.fc.load_state_dict(gcn.fc.state_dict())
    gcn2.fc.requires_grad = False
    e_su, e_si, su, si = gcn()
    e_su = e_su.detach()
    e_si = e_si.detach()
    su = su.detach()
    si = si.detach()
    si_opp = (1 - F.softmax(si, dim=1)) / (classes_num - 1)

    for epoch in tqdm(range(1000)):
        pos_select_list, neg_select_list = sample(pos_all_list, neg_all_list, n_users, args)
        e_su2, e_si2, su2, si2 = gcn2()
        # su2=F.softmax(su2, dim=1)
        classify_loss = compute_cross_entropy(su2, sens_opp) / n_users
        classify_loss = classify_loss * args.c_reg
        sim_loss = calc_sim_loss(e_su2, user_list, pos_select_list, neg_select_list)
        sim_loss = sim_loss * args.s_reg

        # si2 = F.softmax(si2, dim=1)
        diff_loss = compute_cross_entropy(si2, si_opp) / n_items
        diff_loss = diff_loss * args.d_reg

        loss = classify_loss + diff_loss + sim_loss
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()

        final_loss = classify_loss.item()
        diff = diff_loss.item()
        sim = sim_loss.item()


def train_unify_mi(sens_enc, sens_enc2, inter_enc, club, club2, e_xu, e_xi, dataset, u_sens,
                   n_users, n_items, train_u2i, test_u2i, args):
    optimizer_G = optim.Adam(inter_enc.parameters(), lr=args.lr)
    optimizer_D = optim.Adam(club.parameters(), lr=args.lr)
    optimizer_D2 = optim.Adam(club2.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    u_sens = torch.tensor(u_sens).to(torch.long).to(args.device)
    e_su, e_si, _, _ = sens_enc.forward()
    e_su = e_su.detach().to(args.device)
    e_si = e_si.detach().to(args.device)
    e_su2, e_si2, _, _ = sens_enc2.forward()
    e_su2 = e_su2.detach().to(args.device)
    e_si2 = e_si2.detach().to(args.device)

    p_su1 = conditional_samples(e_su.detach().cpu().numpy())
    p_si1 = conditional_samples(e_si.detach().cpu().numpy())
    p_su1 = torch.tensor(p_su1).to(args.device)
    p_si1 = torch.tensor(p_si1).to(args.device)

    p_su2 = conditional_samples(e_su2.detach().cpu().numpy())
    p_si2 = conditional_samples(e_si2.detach().cpu().numpy())
    p_su2 = torch.tensor(p_su2).to(args.device)
    p_si2 = torch.tensor(p_si2).to(args.device)

    best_perf = 0.0

    for epoch in range(args.num_epochs):
        train_res = {
            'bpr': 0.0,
            'emb': 0.0,
            'lb': 0.0,
            'ub': 0.0,
            'mi': 0.0,
            'mi2': 0.0,
            'equal': 0.0,
        }

        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)

            main_user_emb, main_item_emb = inter_enc.forward()

            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg

            e_zu, e_zi = inter_enc.forward()
            lb1 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
                                                    e_su[torch.unique(u)], p_su1[torch.unique(u)])
            lb2 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
                                                    e_si[torch.unique(i)], p_si1[torch.unique(i)])
            lb3 = condition_info_nce_for_embeddings(e_xu[torch.unique(u)], e_zu[torch.unique(u)],
                                                    e_su2[torch.unique(u)], p_su2[torch.unique(u)], alpha=-0.1)
            lb4 = condition_info_nce_for_embeddings(e_xi[torch.unique(i)], e_zi[torch.unique(i)],
                                                    e_si2[torch.unique(i)], p_si2[torch.unique(i)], alpha=-0.1)
            lb = args.lreg * (lb1 + lb2 + lb3 + lb4)

            up1 = club.forward(e_zu[torch.unique(u)], e_su[torch.unique(u)])
            up2 = club2.forward(e_zu[torch.unique(u)], e_su2[torch.unique(u)])

            cos_loss1 = torch.cosine_similarity(e_zu[torch.unique(u)], e_su2[torch.unique(u)], dim=1)
            cos_loss2 = torch.cosine_similarity(e_zu[torch.unique(u)], e_su[torch.unique(u)], dim=1)
            equal_u = torch.mean((cos_loss1 - cos_loss2).abs())

            equal_loss = equal_u * args.equ_reg
            up = (up1 + up2) * args.ureg

            loss = lb + bpr_loss + emb_loss + up + equal_loss

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['lb'] += lb.item()
            train_res['ub'] += up.item()
            train_res['bpr'] += bpr_loss.item()
            train_res['emb'] += emb_loss.item()
            train_res['equal'] += equal_loss.item()

        train_res['bpr'] = train_res['bpr'] / len(train_loader)
        train_res['emb'] = train_res['emb'] / len(train_loader)
        train_res['lb'] = train_res['lb'] / len(train_loader)
        train_res['ub'] = train_res['ub'] / len(train_loader)
        train_res['equal'] = train_res['equal'] / len(train_loader)

        e_zu, e_zi = inter_enc.forward()

        x_samples = e_zu.detach()
        xi_samples = e_zi.detach()
        y_samples = e_su.detach()
        y2_samples = e_su2.detach()
        yi_samples = e_si.detach()
        yi2_samples = e_si2.detach()

        for _ in range(args.train_step):
            mi_loss = club.learning_loss(x_samples, y_samples)
            optimizer_D.zero_grad()
            mi_loss.backward()
            optimizer_D.step()
            train_res['mi'] += mi_loss.item()

        for _ in range(args.train_step):
            mi_loss = club2.learning_loss(x_samples, y2_samples)
            optimizer_D2.zero_grad()
            mi_loss.backward()
            optimizer_D2.step()
            train_res['mi2'] += mi_loss.item()

        train_res['mi'] = train_res['mi'] / args.train_step
        train_res['mi2'] = train_res['mi2'] / args.train_step
        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)

        with torch.no_grad():
            t_user_emb, t_item_emb = inter_enc.forward()
            test_res = ranking_evaluate(
                user_emb=t_user_emb.detach().cpu().numpy(),
                item_emb=t_item_emb.detach().cpu().numpy(),
                n_users=n_users,
                n_items=n_items,
                train_u2i=train_u2i,
                test_u2i=test_u2i,
                sens=u_sens.cpu().numpy(),
                num_workers=args.num_workers)

            p_eval = ''
            for keys, values in test_res.items():
                p_eval += keys + ':' + '[%.6f]' % values + ' '
            print(p_eval)

            if best_perf < test_res['ndcg@10']:
                best_perf = test_res['ndcg@10']
                # torch.save(inter_enc, args.param_path)
                print('save successful')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ml_bpr_fairness',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    torch.set_printoptions(precision=4, edgeitems=10, sci_mode=False, linewidth=160)
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    parser.add_argument('--bakcbone', type=str, default='bpr')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process.pkl')
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--log_path', type=str, default='logs/ml/bprmf/bpr_fairness_gender_' + str(time_stamp) + '.txt')
    parser.add_argument('--param_path', type=str,
                        default='param/ml/bprmf/bpr_fairness_gender_' + str(time_stamp) + '.pth')
    parser.add_argument('--pretrain_path', type=str, default='param/bpr_base.pth')#file path of the pretrained base model
    parser.add_argument('--lreg', type=float, default=0.1)
    parser.add_argument('--ureg', type=float, default=0.1)
    parser.add_argument('--sim_epochs', type=int, default=1000)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_step', type=int, default=50)
    parser.add_argument('--s_reg', type=float, default=1.0)
    parser.add_argument('--c_reg', type=float, default=1.0)
    parser.add_argument('--d_reg', type=float, default=1.0)
    parser.add_argument('--equ_reg', type=float, default=1.0)
    parser.add_argument('--u_threshold', type=float, default=0.65)
    parser.add_argument('--l_threshold', type=float, default=0.30)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    sys.stdout = Logger(args.log_path)
    print(args)

    seed = args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)

    bprmf = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    u_sens = user_side_features['gender'].astype(np.int32)
    dataset = BPRTrainLoader(train_set, train_u2i, n_items)
    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()
    classes_num = np.unique(u_sens).shape[0]
    print(np.unique(u_sens))
    sens_enc = SemiGCN(n_users, n_items, norm_adj,
                       args.emb_size, args.n_layers, args.device,
                       nb_classes=classes_num)
    sens_enc2 = SemiGCN(n_users, n_items, norm_adj,
                        args.emb_size, args.n_layers, args.device,
                        nb_classes=classes_num)
    ex_enc = torch.load(args.pretrain_path)
    e_xu, e_xi = ex_enc.forward()
    e_xu = e_xu.detach().to(args.device)
    e_xi = e_xi.detach().to(args.device)
    inter_enc = BPRMF(n_users, n_items, args.emb_size, device=args.device)
    club1 = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)
    club2 = CLUBSample(args.emb_size, args.emb_size, args.hidden_size, args.device)
    train_semigcn(sens_enc, sens_enc2, u_sens, n_users, n_items, e_xu, e_xi, args, classes_num)
    train_unify_mi(sens_enc, sens_enc2, inter_enc, club1, club2, e_xu, e_xi, dataset, u_sens, n_users,
                   n_items, train_u2i, test_u2i, args)
    sys.stdout = None