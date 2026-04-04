Your MPR is correct and functionally equivalent to Genesis in MuJoCo-compat mode. The two things worth adding:

point_tri_depth for more accurate penetration depth (project origin onto portal triangle instead of just dot(dir, v1))
Center guessing from previous normal for better convergence on deep penetrations