from aicc.board import Board

def test_dumps():
    board = Board()
    board.setup_board(2)
    board.state[5, 4] = 1
    board.state[5, 5] = 1
    board.state[6, 5] = 1
    board.state[7, 4] = 1
    r0, c0 = 2, 4
    highlight_cells = []
    for r in range(Board.GRID_DIM):
        for c in range(Board.GRID_DIM):
            if board.is_valid_move(r0, c0, r, c):
                highlight_cells.append((r, c))
    assert board.dumps(highlight_cells=highlight_cells) == (
"""    1            
    11           
    111          
    1111         
....*.*......    
 ...11.......    
  ..*1*......    
   .1........    
    *........    
    ..........   
    ...........  
    ............ 
    .............
         2222    
          222    
           22    
            2    """)
