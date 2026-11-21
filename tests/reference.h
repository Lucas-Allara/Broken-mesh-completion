// Reference values of the tests. They come from the Python implementation of this pipeline that
// preceded this code (numpy, scipy and the tetgen package 0.8.4, which builds the same TetGen).
// The code follows it step by step, down to the random generator and the order of the sums, so
// that every stage can be compared with it. That implementation is not part of the repository,
// so these values are fixed.
#pragma once

namespace reference {

const double random_0[3] = {0.63696168732145431, 0.26978671376387031, 0.040973523936194689};
const long long integers_0[4] = {440LL, 198LL, 1LL, 912755577277LL};
const int choice_0[8] = {4, 4, 4, 5, 5, 0, 5, 1};
const double random_3[3] = {0.085649167143624361, 0.2368105065960997, 0.80127446520639689};
const long long integers_3[4] = {5090LL, 6985LL, 0LL, 433126940236LL};
const int choice_3[8] = {4, 2, 4, 1, 4, 4, 4, 4};
const double spot_area = 5.709518785165157;
const int holes_faces[3] = {4547, 4731, 4786};
const long long holes_checksum[3] = {73606878LL, 76685031LL, 73047180LL};
const double winding_points[6][3] = {{0, 0, 0}, {0.10000000000000001, 0.20000000000000001, 0.29999999999999999}, {0.29999999999999999, -0.20000000000000001, 0.59999999999999998}, {2, 2, 2}, {-0.20000000000000001, 0.40000000000000002, -0.10000000000000001}, {0, 0.5, 0.90000000000000002}};
const double winding_direct[6] = {0.93278189073579021, 0.95920124089194791, 0.94850744088012084, 0.00057845172679046918, 0.60200616333031909, 8.3828207259816417e-05};
const double winding_hierarchy[6] = {0.9327818907357901, 0.95920124089194791, 0.94850744088012084, 0.00057845172679048392, 0.60200616333031909, 8.3828207259772913e-05};

const int case_faces = 4547, case_tets = 15199, case_pairs = 30087;
const double case_sigma = 0.24653517635815064, case_w_sum = 7409.6911116990123;
const int case_inside_cut = 7950, case_inside_heuristics = 7957;
const int case_method_vertices = 2346, case_method_faces = 4684;
const int case_faired_vertices = 2859, case_faired_faces = 5710;
const double case_faired_sum[3] = {-34.07423323451026, -6.7986598334457486, -81.884249902006303};
const int case_hard_inside = 7954, case_hard_faces = 4688;
const double case_w4_sum = 7405.9857582351251;
const int case_default_inside = 7925;
const double case_default_sigma = 0.21435320806867442;
const double case_samples[2][3] = {{0.16658049852912954, 0.0053282108258198187, -0.51958864636293767}, {0.0018123599959920336, 0.2741967648726662, -0.28828858937467089}};
const double case_axes[3][3] = {{-0.0024009057360624021, -0.76144278424431211, -0.64822767757472133}, {0.098012873660704394, 0.64492921431197414, -0.75793125356044733}, {-0.99518225077038891, 0.065354378959717227, -0.073082781161996888}};
const double case_plane[5] = {-0.99999972621355682, 0.0003673232299722999, -0.0006423756346141293, 9.7317361585569649e-05, 0.003453810966580673};
const int case_mirrored = 235;
const double case_errors[3] = {2.3997996670074362, 1.4319290194255643, 0.38053287723676599};  // method, fairing, symmetry

const int pieces_crossing = 82, pieces_tets = 4285, pieces_inside = 3385, pieces_faces = 1124;

const double rosenbrock_x[2] = {0.99999999918781435, 0.99999999844191922}, rosenbrock_f = 1.0990889519195732e-18;
const int rosenbrock_iterations = 117;
const double percentile_85 = 0.93634938644703403, median_37 = 0.48982457833415849, median_36 = 0.51016699322519521;

}  // namespace reference
