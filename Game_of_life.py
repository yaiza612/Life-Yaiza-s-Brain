#  Game of life
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

#x = np.arange(10)
#print(np.roll(x, 1))
#print(np.roll(x, -1))
#print(np.roll(np.roll(x, 1, axis=0), x, axis=0))

def activation(brain, steps):
    """
     - brain is the array with the initial state.
     - steps is the generations.
    """
    def counting(x, y):
        return np.roll(np.roll(brain, y, axis=0), x, axis=1) # in both axis

    for _ in range(steps):
        # count the number of neighbours in all the directions (just all the combinations of 1 and 0 possibles)
        Y = counting(1, 0) + counting(0, 1) + counting(-1, 0) \
            + counting(0, -1) + counting(1, 1) + counting(-1, -1) \
            + counting(1, -1) + counting(-1, 1)
        # game of life rules
        brain = np.logical_or(np.logical_and(brain, Y ==2), Y==3)
        brain = brain.astype(int)
        yield brain


def my_activation(brain, neuron_types, restart_counter = None):
    """
     - brain is the array with the initial state (1 activated neurons, 0 inactivated neurons)
     - neuron_types is the array with the positions of inhibitory and excitatory neurons (1 is excitatory and 0 is inhibitory)
     - restart_counter take in account that if there are too many activated neurons we should restart to recover the level of calcium
     of our brain
    """
    def count_neurons(matrix, x, y):
        return np.roll(np.roll(matrix, y, axis=0), x, axis=1)  # in both axis

    def counting_neurons():
    # count the number of neighbours in all the directions (just all the combinations of 1 and 0 possibles)
        activated_inhibitory_neurons = \
            np.logical_and(count_neurons(brain, 1, 0), np.logical_not(count_neurons(neuron_types, 1, 0))).astype(int) + \
            np.logical_and(count_neurons(brain, 0, 1), np.logical_not(count_neurons(neuron_types, 0, 1))).astype(int) + \
            np.logical_and(count_neurons(brain, -1, 0), np.logical_not(count_neurons(neuron_types, -1, 0))).astype(int) + \
            np.logical_and(count_neurons(brain, 0, -1), np.logical_not(count_neurons(neuron_types, 0, -1))).astype(int) + \
            np.logical_and(count_neurons(brain, 1, 1), np.logical_not(count_neurons(neuron_types, 1, 1))).astype(int) + \
            np.logical_and(count_neurons(brain, 1, -1), np.logical_not(count_neurons(neuron_types, 1, -1))).astype(int) + \
            np.logical_and(count_neurons(brain, -1, 1), np.logical_not(count_neurons(neuron_types, -1, 1))).astype(int) + \
            np.logical_and(count_neurons(brain, -1, -1), np.logical_not(count_neurons(neuron_types, -1, -1))).astype(int)

        activated_excitatory_neurons = \
            np.logical_and(count_neurons(brain, 1, 0), count_neurons(neuron_types, 1, 0)).astype(int) + \
            np.logical_and(count_neurons(brain, 0, 1), count_neurons(neuron_types, 0, 1)).astype(int) + \
            np.logical_and(count_neurons(brain, -1, 0), count_neurons(neuron_types, -1, 0)).astype(int) + \
            np.logical_and(count_neurons(brain, 0, -1), count_neurons(neuron_types, 0, -1)).astype(int) + \
            np.logical_and(count_neurons(brain, 1, 1), count_neurons(neuron_types, 1, 1)).astype(int) + \
            np.logical_and(count_neurons(brain, 1, -1), count_neurons(neuron_types, 1, -1)).astype(int) + \
            np.logical_and(count_neurons(brain, -1, 1), count_neurons(neuron_types, -1, 1)).astype(int) + \
            np.logical_and(count_neurons(brain, -1, -1), count_neurons(neuron_types, -1, -1)).astype(int)

        number_activated_inhibitory_neurons = np.sum(activated_inhibitory_neurons.ravel())
        number_activated_excitatory_neurons = np.sum(activated_excitatory_neurons.ravel())
        return activated_excitatory_neurons, activated_inhibitory_neurons, number_activated_excitatory_neurons, number_activated_inhibitory_neurons

    if restart_counter == 2:
        brain = np.random.binomial(n=1, p=np.random.randint(56, 89) / 100, size=brain.shape)
        _,_, number_activated_excitatory_neurons, number_activated_inhibitory_neurons = counting_neurons()
        return brain, None, number_activated_excitatory_neurons, number_activated_inhibitory_neurons
    activated_excitatory_neurons, activated_inhibitory_neurons,number_activated_excitatory_neurons, number_activated_inhibitory_neurons  = counting_neurons()
    if restart_counter is not None:
        restart_counter += 1
        return brain, restart_counter, number_activated_excitatory_neurons, number_activated_inhibitory_neurons
    # my rules
    # 1 is a excitatory cell
    # 0 is a inhibitory cell
    # for activated cells basically if the difference between
    # activated excitatories - activated inhibitories is 1 or 2 will be in
    # the next step activate and other way will be inactive
    # for inactivated cells basically if the difference between
    # activated excitatories - activated inhibitories is 2 will be activated in the next step
    difference_celltype_activated = activated_excitatory_neurons - activated_inhibitory_neurons
    threshold = 0.90
    a = np.sum(brain.ravel())
    if np.sum(brain.ravel()) >= threshold * brain.shape[0] * brain.shape[1]:
        brain = np.zeros_like(brain)
        restart_counter = 0
    elif np.sum(brain.ravel())==0:
        restart_counter = 0
    else:
        restart_counter = None

        brain = np.logical_or(np.logical_and(brain, difference_celltype_activated== 1), difference_celltype_activated == 2)
        brain = brain.astype(int)
    return brain, restart_counter, number_activated_excitatory_neurons, number_activated_inhibitory_neurons

# Glider
N = 100
for w in range(10):
    brain = np.random.binomial(n=1, p=np.random.randint(60, 100)/100, size=(N,N)) # random activated and inactivated neurons
    neuron_types = np.random.binomial(n=1, p=0.9, size=(N,N))  # random positions of excitatory and inhibitory neurons

    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title="Life of Yaiza's brain", artist='JustGlowing')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure()
    fig.patch.set_facecolor('black')
    with writer.saving(fig, "Brain_epileptic_"+str(w)+".mp4", 500):
        plt.spy(brain)
        plt.axis('off')
        writer.grab_frame()
        plt.clf()
        restart_counter = None
        list_ex_neurons = []
        list_in_neurons = []
        for x in range(200):
            y, restart_counter, excitatory, inhibitory = my_activation(brain, neuron_types, restart_counter=restart_counter)
            list_in_neurons.append(inhibitory)
            list_ex_neurons.append(excitatory)
            brain = y
            plt.spy(y)
            plt.axis('off')
            writer.grab_frame()
            plt.clf()
    fig, ax = plt.subplots()
    ax.plot([_ for _ in range(200)], [(d+1e-6)/(i+1e-6) for d, i in zip(list_ex_neurons, list_in_neurons)])
    ax.set(xlabel='time (s)', ylabel='Ratio E/I')
    ax.grid()
    fig.savefig("ratio_brain_epileptic_" +str(w)+"_neurons.png")



#example of the logic of my game
# activate cellis with the cellular tyopes
# 8 neighbours
# 5 excitatory - 3 inhibitory = 2 excitatory
# keep activated
# 6 excitatory - 2 inhibitory = 3 excitatory
# keep activated
# 7 excitatory -1 inhibotry = 1 excitatory not enough get inactivated
# 4 excitatory and 4 excitatory nothing happens so keep activated
# 8 excitatory too much get inactivated
# 2 excitatory and 6 inhiboty get inactivated

"""
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Game of life', artist='JustGlowing')
writer = FFMpegWriter(fps=10, metadata=metadata)

fig = plt.figure()
fig.patch.set_facecolor('black')
with writer.saving(fig, "game_of_life.mp4", 200):
    plt.spy(brain)
    plt.axis('off')
    writer.grab_frame()
    plt.clf()
    for x in activation(brain, 800):
        a = plt.spy(x)
        plt.axis('off')
        writer.grab_frame()
        plt.clf()"""
