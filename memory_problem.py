import torch
import numpy as np
import matplotlib.pyplot as plt
from src.adding_memory_problems import generate_data_memory
from src.rnn import SimpleRecurrentCell, RecurrentLayer, ComplexCell
from src.opt import StiefelOptimizer


if __name__ == '__main__':
    # TODO: Add an argparse
    n_train = int(10e5)
    n_test = int(1e4)
    time_steps = 1
    n_sequence = 10
    train_data = generate_data_memory(time_steps, n_train, n_sequence)
    test_data = generate_data_memory(time_steps, n_test, n_sequence)
    # --- baseline ----------------------
    baseline = np.log(8) * 10/(time_steps + 20)
    print("Baseline is " + str(baseline))
    batch_size = 100
    lr = 0.001 #2e-2
    # cell = ComplexCell(hidden_size=128, input_size=10)
    cell = SimpleRecurrentCell(hidden_size=128, input_size=10)
    # cell = CustomGRU(hidden_size=64, input_size=10)
    # cell.reset_parameters()
    # cell = torch.nn.GRUCell(hidden_size=256, input_size=10)
    model = RecurrentLayer(cell=cell, output_size=10)
    optimizer = StiefelOptimizer(model.parameters(), lr=lr)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    # cost = torch.nn.MSELoss()

    train_x, train_y = generate_data_memory(time_steps, n_train, n_sequence)

    train_x_lst = np.array_split(train_x, n_train//batch_size, axis=0)
    train_y_lst = np.array_split(train_y, n_train//batch_size, axis=0)

    iterations = len(train_x_lst)
    assert len(train_x_lst) == len(train_y_lst)

    loss_lst = []
    norm_lst = []
    # train cell
    for i in range(iterations):
        xx = train_x_lst[i]
        yy = train_y_lst[i]

        x_one_hot = np.zeros([batch_size, 20+time_steps, n_sequence])
        y_one_hot = np.zeros([batch_size, 20+time_steps, n_sequence])
        # one hote encode the inputs.
        for b in range(batch_size):
            for t in range(20+time_steps):
                x_one_hot[b, t, xx[b, t]] = 1
                y_one_hot[b, t, yy[b, t]] = 1
                

        # x = np.expand_dims(x_one_hot, -1)
        # y = np.expand_dims(y_one_hot, -1)
        
        x = torch.from_numpy(x_one_hot.astype(np.float32))
        y = torch.from_numpy(y_one_hot.astype(np.float32))
        yyt = torch.from_numpy(yy.astype(np.float32))
        optimizer.zero_grad()
        out_lst = []
        h_lst = []
        # forward
        # with torch.autograd.set_detect_anomaly(True):
        if 1:
            zero_state =  model.zero_state(batch_size)
            out_tensor = model.forward(input_series=x,
                                       steps=time_steps+20,
                                       init_state=zero_state)
            out_tensor = torch.sigmoid(out_tensor)
            # loss = cost(target=y, input=out_tensor)   
            loss = torch.mean(-y*torch.log(out_tensor + 1e-8)
                              - (1-y)*torch.log(1-out_tensor + 1e-8))
            loss.backward(retain_graph=True)
            optimizer.step()

        loss_lst.append(loss.item())
        # compute accuracy
        y_net = np.squeeze(np.argmax(out_tensor.detach().numpy(), axis=2))
        mem_net = y_net[:, -10:]
        mem_y = yy[:, -10:]
        acc = np.sum((mem_y == mem_net).astype(np.float32))
        acc = acc/(batch_size * 10.)

        if i % 25 == 0:
            print(i, "{:2.2f}".format(loss.item()), "{:2.2f}".format(acc), "{:2.2f}".format(i/iterations))
        if i % 250 == 0:
            print(yy[0, :])
            print(y_net[0, :])
            if type(cell) is SimpleRecurrentCell or type(cell) is ComplexCell:
                try:
                    norm = torch.linalg.norm(cell.weight_recurrent, ord=2).item()
                    print('norm {:2.2f}'.format(norm))
                    norm_lst.append(norm)
                except:
                    print('norm', 'nan')


plt.semilogy(loss_lst, label='loss')
plt.show()

plt.title('orthogonality error')
plt.semilogy(np.abs(1 - np.stack(norm_lst)))
plt.show()
print('done')