import sam.lsam as lsam
import sam.lsam_wn as lsam_wn
import sam.qlsam as qlsam
import sam.qlsamv2_wn as qlsamv2_wn
import sam.sam as sam


def get_minimizer(model, optimizer, args):
    if "QLSAMv2" in args.sam_type:
        minimizer = qlsamv2_wn.QSAM(
            optimizer,
            model,
            rho=args.rho,
            include_norm=args.include_norm,
        )
    elif "QLSAM" in args.sam_type:
        minimizer = qlsam.QSAM(
            optimizer,
            model,
            rho=args.rho,
            include_norm=args.include_norm,
        )
    elif "LSAM_wn" in args.sam_type:
        minimizer = lsam_wn.SAM(
            optimizer,
            model,
            rho=args.rho,
            include_norm=args.include_norm,
        )
    elif "LSAM" in args.sam_type:
        minimizer = lsam.SAM(
            optimizer,
            model,
            rho=args.rho,
            include_norm=args.include_norm,
        )
    elif "SAM" in args.sam_type:
        minimizer = sam.SAM(
            optimizer,
            model,
            rho=args.rho
        )
    return minimizer
