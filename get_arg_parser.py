"""
https://github.com/chris-chris/mlops-example
"""
import argparse


def get_params():
    """
    Parse args params
    :return:
    """
    parser = argparse.ArgumentParser()

    # data hyperparams
    parser.add_argument("inputs", action='store', nargs='*', help='help 10')
    parser.add_argument("--input_size", type=int, default=10)
    parser.add_argument("--refit", action='store_true')
    parser.add_argument("-d", action='store_true')

    parser.add_argument("--framework", type=str, default="sklearn")
    parser.add_argument("--keras_model", type=str, default="dense")
    parser.add_argument("--sklearn_model", type=str, default="linear")
    parser.add_argument("--loss", type=str, default="squared_loss")

    # training hyperparams

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()

    return args


# @ex.automain
# def run(args):
#     """
#     Run sacred experiment
#     :param args:
#     :return:
#     """
#     if args.framework == 'sklearn':
#         test_loss = sk.train_sklearn(args)
#     elif args.framework == 'keras':
#         test_loss = ke.train_keras(args, ex)
#     else:
#         return None

#     ex.log_scalar('loss', test_loss)

#     return test_loss


if __name__ == "__main__":
    args = get_params()
    print(args)
    try:
        import IPython
        print( "Suggestions:\n")
        # print("%%timeit get_params(%d, '%s')" % (i, s))
        print(args)
        IPython.embed()
    except:
        pass
