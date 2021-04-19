from utility import load_dataset, PLSInstance
import argparse

########################################################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename", type=str, default=None, required=True,
                        help="Path of the txt file with the partial solutions - assignments pairs")
    parser.add_argument("--partial-sols-filename", type=str, default=None, required=True,
                        help="Path where the partial solutions are saved to")
    parser.add_argument("--domains-filename", type=str, default=None, required=False,
                        help="Path where the variables domains are saved to")
    parser.add_argument("--domains-type", choices=['full', 'rows'], default=None,
                        help="Compute variables domains with forward checking propagator and save them in a CSV file.")
    parser.add_argument("--assignments-filename", type=str, default=None, required=True,
                        help="Path where the assignments are saved to")
    parser.add_argument("--dim", type=int, default=None, required=True,
                        help="Problem dimension")

    args = parser.parse_args()

    leave_columns_domains = False
    if args.domains_type == 'rows':
        leave_columns_domains = True

    load_dataset(filename=args.filename,
                 problem=PLSInstance(n=args.dim, leave_columns_domains=leave_columns_domains),
                 mode="onehot",
                 save_domains=args.save_domains,
                 domains_filename=args.domains_filename,
                 save_partial_solutions=True,
                 partial_sols_filename=args.partial_sols_filename,
                 assignments_filename=args.assignments_filename)