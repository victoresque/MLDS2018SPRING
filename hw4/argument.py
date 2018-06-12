def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--save-freq', type=int, default=10,
                        help='saving frequency')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for reward in training')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate for training')
    parser.add_argument('--hidden-size', type=int, default=200,
                        help='hidden size for the training model')
    return parser
