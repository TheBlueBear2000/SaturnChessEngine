import server.SaturnChessEngine.saturn_shared as saturn_shared
from server.SaturnChessEngine.network import SaturnNetwork
from server.SaturnChessEngine.util import mask_to_bitmap

import json
import torch
from time import sleep


class Engine:
    def __init__(self):
        self.board = saturn_shared.Board()

        with open("server/SaturnChessEngine/allowed_moves.json", "r") as file:
            self.moves_list = json.load(file)

        # Number of outputs is equal to the number of moves listed in allowed_moves
        self.model = SaturnNetwork(len(self.moves_list))
        self.model.load_state_dict(
            torch.load("server/SaturnChessEngine/saturn_weights.pth")
        )  # Load weights
        self.model.eval()  # Inference mode

    def playMove(self, move):
        # sleep(4)
        print(f"\nSATURN: Updated by playing move: {move}")
        self.board.update(move)
        print(self.board.render())

    def decideMove(self):
        board = []
        # Translate bitboard numbers into 3D bitmap
        for channel in self.board.state:
            board.append(mask_to_bitmap(channel))
        board = torch.tensor(board, dtype=torch.float32)
        board = board.unsqueeze(0)

        board_params = self.board.getParams()
        params = board_params["castling_rights"] + board_params["scores"]
        params = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        move_logits, promotion_logits, _ = self.model(board, params)
        sorted_moves = torch.argsort(move_logits[0], descending=True)

        moves_discarded = 0
        for move_index in sorted_moves:
            move = self.moves_list[move_index.item()]
            legal, reason = self.board.moveLegal(move)
            if legal:  # Move is legal

                # Add promotion if needed
                startPos = saturn_shared.Position(move[0], move[1])
                channel_at_pos = self.board.identifyChannelFromPos(startPos)
                if (channel_at_pos == 0 and move[3] == 8) or (
                    channel_at_pos == 6 and move[3] == 1
                ):
                    # Select promotion
                    move += ["q", "n", "b", "r"][
                        torch.argmax(promotion_logits[0]).item()
                    ]

                # Actually, do not perform update here. Update is sent back from server
                # Perform move to update board
                # self.board.update(move)
                print(
                    f"\nSATURN: Decided on move: {move}, type {type(move)}. Discarded {moves_discarded} moves"
                )
                return move
            else:
                moves_discarded += 1
                # Attempted illegal moves can be logged here later
                # print(f"Wanted to do move {move}, but couldn't: {reason}")
