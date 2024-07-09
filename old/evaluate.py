import old_nueral_net as nun
import sys
import csv
from utils import get_col_names
# In this box are the params that need to be checked before running
#-----------------------------------------------------------------#
# True for eval, false for training
EVALUATING = True
# Hyperparameters
epochs = 100
learning_rate = 0.01

# Loss Function
# Where is this?
loss_fn = nun.torch.nn.MSELoss()

# Quality of life parameters - this is in the nn file rn. 
# Id like to pull it out if possible, but I don't know how
# batch_print_frequency = 10000

# Choose which model to use
cur_model = nun.good_baseline
cur_model_name = nun.model_names["fast_model"]
cur_model_set = nun.dataset_names["full_testing"]
# basic, skill, single, or winrate so far
eval_name = "single"
cur_evaluator = nun.eval_names[eval_name]
#-----------------------------------------------------------------#
def which_evaluator(model, data, name, isgood, eval_name):
    if eval_name == 'basic':
        nun.evaluate(model, data, name)
    elif eval_name == 'winrate':
        nun.evaluate_winrate(model, data, name)
    elif eval_name == 'skill':
        nun.evaluate_skill_comparison(model, data, name, isgood)
    elif eval_name == 'single':
        nun.evaluate_single(model, data, name)

def nntest(evaluating, data=None):
    if(data==None):
        data = nun.rd.WinrateDataset(cur_model_set, "winrate") if eval_name == "winrate" else nun.rd.DraftDataset(cur_model_set)
    print(data)
    model = nun.NeuralNetwork(net_structure=cur_model).to(nun.device)

    if evaluating: 
        which_evaluator(model, data, cur_model_name, isgood=(cur_model_name == nun.dataset_names["good_player"]), eval_name=cur_evaluator)
        # for i, p in enumerate(model.parameters()):
        #     print(f"{i}: {p}")
        #nun.evaluate_skill_comparison()
        return
    else:
        training_data = nun.rd.WinrateDataset(nun.dataset_names["winrate_training"], 'winrate')
        training_dataloader = nun.rd.DataLoader(training_data, shuffle=True)

        testing_data = nun.rd.WinrateDataset(nun.dataset_names["winrate_testing"], 'winrate')
        testing_dataloader = nun.rd.DataLoader(testing_data)

        optimizer = nun.torch.optim.SGD(model.parameters(), lr=learning_rate)
        # Optionally continue where you left off
        #model.load_state_dict(nun.torch.load("model_weights.pth"))

        for e in range(epochs):
            print(f"Epoch {e+1}\n------------- ------------------")
            nun.train_loop(training_dataloader, model, loss_fn, optimizer)
            nun.test_loop(testing_dataloader, model, loss_fn)
        print("Done!")
        nun.torch.save(model.state_dict(), 'model_weights.pth')
        print("Saved!")

def main():
    # data = read_input_pack() if eval_name == "single" else None
    nntest(EVALUATING, data=None)

if __name__ == "__main__":
    main()