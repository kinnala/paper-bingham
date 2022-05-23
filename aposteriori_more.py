from skfem import *
from skfem.helpers import *
from skfem.visuals.matplotlib import *
from scipy.sparse import bmat
#from adaptmesh import triangulate
from scipy.sparse.linalg import splu

import numpy as np

# Glowinski-Lions-Tremolieres (1981) p. 354
R = 1  # radius of the domain
f = 3.6
g = 1.25  # g < f * R / 2 to get nonzero solution
#g = 1.25
Rp = 2 * g / f


#m = MeshTri.init_circle(nrefs=3)
#m = MeshTri.load("/home/tom/circle3.msh").smoothed()
#m = MeshTri.init_circle(nrefs=1).refined(4)
#m = MeshTri.init_circle(nrefs=0).scaled(0.7).refined(4)
# m = triangulate([(0.1, 0.1),
#                  (0.5, -0.2),
#                  (1, 0),
#                  (1.2, 0.5),
#                  (1, 1),
#                  (0.5, 1.1),
#                  (0, 0.5)]).refined(3)
#m = MeshTri.init_refdom().refined(4)
#m = MeshTri.init_sqsymmetric().scaled(1.3).refined(4)

casename = 'square'
#casename = 'lshaped'

if casename == 'square':
    m = MeshTri.init_sqsymmetric().scaled(2.).translated((-1., -1.)).refined(4)
elif casename == 'lshaped':
    m = MeshTri.init_lshaped().refined(4)


#m = MeshTri.init_sqsymmetric().scaled(0.82).refined(4)
#xs = np.geomspace(0.5, 1.0, 20) - 0.5
#xs = np.concatenate((xs, -xs[:-1] + 1.))
#m = MeshTri.init_tensor(xs, xs).smoothed()
#m = triangulate([(0, 0), (1, 0), (1, 1), (0, 1)]).refined(3).smoothed()
#m = triangulate([(0, 0), (1, 0), (1, 1), (0, 1)]).scaled(1.3).refined(3)
#m = MeshTri.init_tensor(
#   np.linspace(-.5, .5, 41),
#   np.linspace(-1, 1, 41),
#)
#m = MeshTri.init_lshaped().scaled(2.0).refined(4)
#uniform = True
uniform = False

for itr in range(15 if not uniform else 4):
    if itr > 0:
        if uniform:
            m = m.refined()
        else:
            m = m.refined(elems)

    if not uniform and itr > 0:
        m = m.smoothed()

    m.draw()
    case = casename
    plt.savefig('aposteriori_more_{}_mesh_{}.pdf'.format(case, itr))

    e, e00, ename = [
        (ElementTriMini(), ElementTriP1(), "p1b_p1"),
        (ElementTriP2(), ElementTriP0(), "p2_p0"),
        (ElementTriP1(), ElementTriP0(), "p1_p0"),
        (ElementTriP3(), ElementDG(ElementTriP1()), "p3_p1dg"),
    ][3]
    case += ename
    if uniform:
        case += "_uniform"
    else:
        case += "_adaptive"

    e0 = ElementVector(e00)
    egradu = ElementVector(ElementDG(ElementTriP2()))
    basis = Basis(m, e)
    basis0 = basis.with_element(e0)
    basis00 = basis.with_element(e00)
    basisgradu = basis.with_element(egradu)

    # r < Rp
    ui = (R - Rp) / 2. * (f / 2. * (R + Rp) - 2. * g)

    # Rp < r < R
    r = lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2)
    rx = lambda x: x[0] / r(x)
    ry = lambda x: x[1] / r(x)

    uo = lambda x: (R - r(x)) / 2. * (f / 2. * (R + r(x)) - 2. * g)
    uox = lambda x: (-rx(x) / 2. * (f / 2. * (R + r(x)) - 2. * g)
                     + (R - r(x)) / 2. * (f / 2. * rx(x)))
    uoy = lambda x: (-ry(x) / 2. * (f / 2. * (R + r(x)) - 2. * g)
                     + (R - r(x)) / 2. * (f / 2. * ry(x)))
    uoxx = lambda x: (-1/2*f*x[0]**2*((1/2)*R - 1/2*np.sqrt(x[0]**2 + x[1]**2))/(x[0]**2 + x[1]**2)**(3/2) - 1/2*f*x[0]**2/(x[0]**2 + x[1]**2) + (1/2)*f*((1/2)*R - 1/2*np.sqrt(x[0]**2 + x[1]**2))/np.sqrt(x[0]**2 + x[1]**2) + (1/2)*x[0]**2*((1/2)*f*(R + np.sqrt(x[0]**2 + x[1]**2)) - 2*g)/(x[0]**2 + x[1]**2)**(3/2) - 1/2*((1/2)*f*(R + np.sqrt(x[0]**2 + x[1]**2)) - 2*g)/np.sqrt(x[0]**2 + x[1]**2))
    uoyy = lambda x: (-1/2*f*x[1]**2*((1/2)*R - 1/2*np.sqrt(x[0]**2 + x[1]**2))/(x[0]**2 + x[1]**2)**(3/2) - 1/2*f*x[1]**2/(x[0]**2 + x[1]**2) + (1/2)*f*((1/2)*R - 1/2*np.sqrt(x[0]**2 + x[1]**2))/np.sqrt(x[0]**2 + x[1]**2) + (1/2)*x[1]**2*((1/2)*f*(R + np.sqrt(x[0]**2 + x[1]**2)) - 2*g)/(x[0]**2 + x[1]**2)**(3/2) - 1/2*((1/2)*f*(R + np.sqrt(x[0]**2 + x[1]**2)) - 2*g)/np.sqrt(x[0]**2 + x[1]**2))

    # combine
    uanal = lambda x: (r(x) > Rp) * uo(x) + (r(x) < Rp) * ui
    uanalx = lambda x: (r(x) > Rp) * uox(x)
    uanaly = lambda x: (r(x) > Rp) * uoy(x)
    uanalxx = lambda x: (r(x) > Rp) * uoxx(x)
    uanalyy = lambda x: (r(x) > Rp) * uoyy(x)

    divlamanal = lambda x: ((r(x) < Rp) * (-f / g)
                            + (r(x) > Rp) * (-f - uanalxx(x) - uanalyy(x)) / g)

    @BilinearForm
    def laplace(u, v, w):
        x, y = w.x
        return dot(grad(u), grad(v))

    @BilinearForm
    def constraint(lam, v, w):
        return g * dot(lam, grad(v))

    @LinearForm
    def rhs(v, w):
        return f * v

    A = asm(laplace, basis)
    B = asm(constraint, basis0, basis)
    b = asm(rhs, basis)

    D = basis.get_dofs().all(['u'])
    I = basis.complement_dofs(D)
    Af = splu(A[I].T[I])

    u = basis.zeros()
    lam = basis0.zeros()

    maxiters = 5000
    ress = []
    # contact iteration
    for itr in range(maxiters):

        lamxdofs = basis0.get_dofs(elements=lambda x: 1 + x[0] * 0).all('u^1')
        lamydofs = basis0.get_dofs(elements=lambda x: 1 + x[0] * 0).all('u^2')

        eps = 1.5

        if 1:
            ux = basis00.project(basis.interpolate(u).grad[0])
            uy = basis00.project(basis.interpolate(u).grad[1])

            sx = lam[lamxdofs] + eps * ux
            sy = lam[lamydofs] + eps * uy

            length = np.sqrt(sx ** 2 + sy ** 2)

            lam[lamxdofs] = sx / np.maximum(1, length)
            lam[lamydofs] = sy / np.maximum(1, length)
        else:
            ux = basis.interpolate(u).grad[0]
            uy = basis.interpolate(u).grad[1]
            lamx = basis0.interpolate(lam).value[0]
            lamy = basis0.interpolate(lam).value[1]

            sx = lamx + eps * ux
            sy = lamy + eps * uy

            length = np.sqrt(sx ** 2 + sy ** 2)

            lam[lamxdofs] = basis00.project(sx / np.maximum(1, length))
            lam[lamydofs] = basis00.project(sy / np.maximum(1, length))

        print(np.sum(np.sqrt(lam[lamxdofs] ** 2 + lam[lamydofs] ** 2) > 1 + 1e-4))

        # enforce constraint
        uprev = u.copy()
        u[I] = Af.solve(b[I] - B.dot(lam)[I] - A[I].T[D].T.dot(u[D]))

        res = np.linalg.norm(u - uprev) / np.linalg.norm(u)
        ress.append(res)
        print("Residual: {}, iteration: {}".format(res, itr))
        if res < 1e-7:
            break

    # calculate a posteriori estimators
    @Functional
    def residual_est(w):
        lam = w['lam']
        gradu = w['gradu']
        lapu = gradu.grad[0, 0] + gradu.grad[1, 1]
        return w.h ** 2 * (lapu + g * div(lam) + f) ** 2

    @Functional
    def nc_est(w):
        # lam = w['lam']
        # u = w['u']
        # return g * (np.sqrt(dot(grad(u), grad(u))) - dot(lam, grad(u)))
        lam = w['lam']
        u = w['u']
        pgradu = w['pgradu']
        length_lam = np.sqrt(dot(lam, lam))
        length_gradu = np.sqrt(dot(grad(u), grad(u)))
        length_pgradu = np.sqrt(dot(pgradu, pgradu))
        return g * (length_gradu
                    - lam[0] / np.maximum(length_lam, 1) * pgradu[0]
                    - lam[1] / np.maximum(length_lam, 1) * pgradu[1])

    @Functional
    def jump_est(w):
        lam0 = w['lam0']
        gradu0 = w['gradu0']
        lam1 = w['lam1']
        gradu1 = w['gradu1']
        return w.h * dot(gradu0 + g * lam0 - (gradu1 + g * lam1), w.n) ** 2

    gradu = basisgradu.project(basis.interpolate(u).grad)
    apost1 = residual_est.elemental(basis,
                                    lam=basis0.interpolate(lam),
                                    gradu=basisgradu.interpolate(gradu))
    m.plot(apost1, colorbar=True)
    plt.title('$\eta_T$')

    fbasis0_0 = InteriorFacetBasis(basis0.mesh, basis0.elem, side=0, intorder=4)
    fbasis0_1 = InteriorFacetBasis(basis0.mesh, basis0.elem, side=1, intorder=4)
    fbasis_0 = InteriorFacetBasis(basisgradu.mesh, basisgradu.elem, side=0, intorder=4)
    fbasis_1 = InteriorFacetBasis(basisgradu.mesh, basisgradu.elem, side=1, intorder=4)
    _apost2 = jump_est.elemental(fbasis0_0,
                                 lam0=fbasis0_0.interpolate(lam),
                                 lam1=fbasis0_1.interpolate(lam),
                                 gradu0=fbasis_0.interpolate(gradu),
                                 gradu1=fbasis_1.interpolate(gradu))

    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis_0.find, _apost2)
    apost2 = np.sum(.5 * tmp[m.t2f], axis=0)
    m.plot(apost2, colorbar=True)
    plt.title('$\sum_{E \subset \partial K} \eta_E$')

    # eta con
    ux = basis00.project(basis.interpolate(u).grad[0])
    uy = basis00.project(basis.interpolate(u).grad[1])
    sx = lam[lamxdofs] + eps * ux
    sy = lam[lamydofs] + eps * uy
    sxy = lam.copy()
    sxy[lamxdofs] = sx
    sxy[lamydofs] = sy

    apost3 = nc_est.elemental(basis,
                              lam=basis0.interpolate(sxy),
                              u=basis.interpolate(u),
                              pgradu=basis0.interpolate(basis0.project(basis.interpolate(u).grad)))
    # apost3 = nc_est.elemental(basis,
    #                           lam=basis0.interpolate(lam),
    #                           u=basis.interpolate(u))
    m.plot(apost3, colorbar=True)
    plt.title('$\eta_{con}$')

    elems = adaptive_theta(np.sqrt(apost1 + apost2 + apost3), theta=0.5)

    errors = "{},{}\n".format(
        len(u) + len(lam),
        np.sqrt(np.sum(apost1 + apost2 + apost3))
    )
    print(errors)
    with open("aposteriori_more_{}.csv".format(case), "a+") as handle:
        handle.write(errors)

    if len(u) + len(lam) > 175000:
        break
    basis.plot(u, colorbar={'orientation': 'horizontal'}, shading='gouraud')
    lamx = lam[lamxdofs]
    lamy = lam[lamydofs]
    plot(
        basis00,
        np.sqrt(lamx ** 2 + lamy ** 2),
        nrefs=1,
        figsize=(8, 8),
        shading='gouraud',
        levels=[0.1, 0.3, 0.5, 0.7, 0.9],
        colorbar={'orientation': 'horizontal'},
    )
    plt.close('all')
    #print(np.max(u))
    #plt.show()


# plotting
import matplotlib.pyplot as plt

plt.semilogy(np.array(ress))
plt.show()

ux = basis00.project(basis.interpolate(u).grad[0])
uy = basis00.project(basis.interpolate(u).grad[1])

ax = draw(m)
# plot(pbasis, p, shading='gouraud', colorbar=True, ax=ax)
# plt.title('$p$')
plot(basis, u, nrefs=1, figsize=(8, 8), shading='gouraud', colorbar={'orientation': 'horizontal'}, levels=5)
#plt.show()
plt.savefig('aposteriori_more_{}_u.pdf'.format(case))
#plt.title('$u$')
ax = draw(m)
plot(basis00, np.sqrt(ux ** 2 + uy ** 2), nrefs=0, figsize=(8, 8), colorbar=True, ax=ax)
plt.title('$\|\\nabla u\|$')

uanalp = basis.project(uanal)

plot(basis, u - uanalp, nrefs=1, shading='gouraud', figsize=(8, 8), colorbar=True)
plot(basis, uanalp, nrefs=1, shading='gouraud', figsize=(8, 8), colorbar=True)

# draw lambda
lamx = lam[lamxdofs]
lamy = lam[lamydofs]
plot(
    basis00,
    np.sqrt(lamx ** 2 + lamy ** 2),
    nrefs=1,
    figsize=(8, 8),
    shading='gouraud',
    #levels=[0.1, 0.3, 0.5, 0.7, 0.9],
    colorbar={'orientation': 'horizontal'},
)
#plt.show()
plt.savefig('aposteriori_more_{}_lam.pdf'.format(case))
#plt.title('$\|\lambda\|$')
plot(
    basis00,
    lamy,
    nrefs=1,
    figsize=(8, 8),
    shading='gouraud',
    colorbar={'orientation': 'horizontal'},
)
plt.title('$\lambda_y$')


# draw weak div(lambda)
# from skfem.models import mass

# M = mass.assemble(basis)
# divlam = solve(M, B.dot(lam))
# plot(basis, divlam, nrefs=3, colorbar=True)

show()
