from game.game import Game
from model import ACTIONS, build_model, next_action_index, reshape_input

if __name__ == "__main__":
    model = build_model()
    model.load_weights("model.h5")
    game = Game()

    game.reset()
    score = game.score
    state = game.game_state()

    while True:
        game.clock.tick(1)
        action_index = next_action_index(model, reshape_input(state), 1.0)
        next_move = ACTIONS[action_index]
        state, score, is_game_over, _ = game.run_action(next_move)

        if is_game_over:
            break
