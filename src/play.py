import statistics
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from cnn_model.constants import FEATURES, WEIGHTS_FILE_NAME
from cnn_model.model import build_model
from cnn_model.train import get_initial_state
from game.constants import COLS, ROWS
from game.game import Game, Player
from game.snake import Direction
from nn_model import DQN_Agent


def user_play():
    game = Game(Player.Human)
    game.play_game()


def model_play(player: Player, weights_path: str, use_graphics: bool = False):
    if player is Player.NN:
        dqn_agent = DQN_Agent(epsilon=0.0)
        dqn_agent.model.load_weights(weights_path)
    elif player is Player.CNN:
        model = build_model()
        model.load_weights(weights_path)
    else:
        raise Exception("No such player: {}".format(player))

    game = Game(player, use_graphics)
    scores = []
    rounds = 100
    progress_bar = tqdm(total=rounds)
    for _ in range(rounds):
        if player is Player.NN:
            game.reset()
            current_state = game.game_state()
        elif player is Player.CNN:
            current_state = get_initial_state(game)

        current_score = 0
        uneventful_rounds = 0
        while True:
            if use_graphics:
                game.clock.tick(30)

            if player is Player.NN:
                next_move = dqn_agent.next_action(  # type: ignore
                    dqn_agent.reshape_input(current_state)  # type: ignore
                )
            elif player is Player.CNN:
                predictions = model.predict(current_state)[0]  # type: ignore
                action_index = np.argmax(predictions)
                next_move = Direction(action_index)

            new_state, _, game_over, _ = game.run_action(next_move)

            if player is Player.NN:
                current_state = new_state
            elif player is Player.CNN:
                # Reshape state
                new_state = np.reshape(new_state, (1, ROWS, COLS, 1, FEATURES))
                # Add new game frame to the next state, delete last (oldest) state
                current_state = np.append(current_state, new_state, axis=3)
                current_state = np.delete(current_state, 0, axis=3)

            if game_over:
                scores.append(game.score)
                break

            uneventful_rounds += 1
            if game.score > current_score:
                current_score = game.score
                uneventful_rounds = 0

            # After 300 rounds of not increasing score, assume we are in a loop, just break loop here
            if uneventful_rounds > 300:
                scores.append(game.score)
                break

        progress_bar.update()

    print("High-score:", game.highscore)
    print("Median score:", statistics.median(scores))
    print("Mean score:", statistics.mean(scores))


if __name__ == "__main__":
    argument_parser = ArgumentParser("Play snake game")

    argument_parser.add_argument(
        "-p",
        "--player",
        help="Type of player (model or human)",
        type=Player,
        choices=list(Player),
    )

    argument_parser.add_argument(
        "-g", "--graphics", help="Use graphics for the model", action="store_true"
    )

    argument_parser.add_argument(
        "-w",
        "--weights-path",
        help="Path to the weights path",
        default=WEIGHTS_FILE_NAME,
    )

    args = argument_parser.parse_args()

    if args.player is not Player.Human:
        model_play(args.player, args.weights_path, args.graphics)
    else:
        user_play()
