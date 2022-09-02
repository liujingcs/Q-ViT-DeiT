import sam.qlsam as qlsam

def get_minimizer(model, optimizer, args):
    if "QLSAM" in args.sam_type:
        minimizer = qlsam.QSAM(
            optimizer,
            model,
            rho=args.rho,
            include_norm=args.include_norm,
        )
    return minimizer
