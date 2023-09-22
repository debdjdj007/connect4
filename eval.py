import os.path
import torch
import numpy as np
import dal2
import copy
import pickle
import datetime
import matplotlib.pyplot as plt
import random

def play_game_auto(net):
    # Randomly decide if the AI plays as "O" or "X"
    play_as = random.choice(["O", "X"])
    if play_as == "O":
        black = net
        white = None
    else:
        white = net
        black = None

    current_board = ConnectBoard()
    checkmate = False
    dataset = []
    value = 0; t = 0.1; moves_count = 0
    while checkmate == False and current_board.moves() != []:
        if moves_count <= 5:
            t = 1
        else:
            t = 0.1
        moves_count += 1
        dataset.append(copy.deepcopy(encodeB(current_board)))
        if current_board.player == 0:
            if white != None:
                root = uctSearch(current_board,777,white,t)
                policy = getPol(root, t)
            else:
                col = random.choice(range(7))
                policy = np.zeros([7], dtype=np.float32); policy[col] += 1
        elif current_board.player == 1:
            if black != None:
                root = uctSearch(current_board,777,black,t)
                policy = getPol(root, t)
            else:
                col = random.choice(range(7))
                policy = np.zeros([7], dtype=np.float32); policy[col] += 1

        current_board = decodeMove(current_board, np.random.choice(np.array([0,1,2,3,4,5,6]), p = policy))

        if current_board.winner() == True:
            if current_board.player == 0:
                value = -1
            elif current_board.player == 1:
                value = 1
            checkmate = True

    dataset.append(encodeB(current_board))
    if value == -1:
        if play_as == "O":
            dataset.append(f"AI as black wins")
        else:
            dataset.append(f"Random as black wins")
        return "black", play_as, dataset
    elif value == 1:
        if play_as == "O":
            dataset.append(f"Random as white wins")
        else:
            dataset.append(f"AI as white wins")
        return "white", play_as, dataset
    else:
        dataset.append("Nobody wins")
        return None, play_as, dataset

def main():
    # List of model filenames
    model_filenames = [f"cc4_current_net__iter{i}.pth.tar" for i in range(0, 7)]

    # Dictionary to hold AI wins for each model
    ai_wins_count = {}

    for model_file in model_filenames:
        model_path = os.path.join("./model_data/", model_file)
        cnet = ConnectionNetwork()
        cuda = torch.cuda.is_available()
        if cuda:
            cnet.cuda()
        cnet.eval()
        checkpoint = torch.load(model_path)
        cnet.load_state_dict(checkpoint['state_dict'])

        # Play game 10 times and count AI wins
        ai_wins = 0
        for _ in range(10):
            winner, ai_played_as, _ = play_game_auto(cnet)
            if (winner == "black" and ai_played_as == "O") or (winner == "white" and ai_played_as == "X"):
                ai_wins += 1

        ai_wins_count[model_file] = ai_wins

    # Plotting the results
    plt.bar(ai_wins_count.keys(), ai_wins_count.values())
    plt.xlabel('Model Filename')
    plt.ylabel('Number of AI Wins')
    plt.title('AI Wins for Each Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()

    # Save the graph to a jpg file
    file_name = f"ai_wins_graph_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    plt.savefig(file_name)
    plt.close()

    return file_name  # Return the saved file name for reference


if __name__ == "__main__":
    main()
