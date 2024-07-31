# class Laminate:
#     def __init__(
#         self,
#         plies: list[Lamina, ...],
#         Nx: float = 0,
#         Ny: float = 0,
#         Ns: float = 0,
#         Mx: float = 0,
#         My: float = 0,
#         Ms: float = 0,
#         midplane=True,
#         load_angle=0,
#     ):
#         self.plies = plies
#         self.Nx = Nx
#         self.Ny = Ny
#         self.Ns = Ns
#         self.Mx = Mx
#         self.My = My
#         self.Ms = Ms
#         self.midplane = midplane
#         self.n_plies = len(self.plies)
#         self.A = np.zeros([3, 3])
#         self.B = np.zeros([3, 3])
#         self.D = np.zeros([3, 3])
#         self.z = np.zeros(self.n_plies + 1)
#         self.ABD = self.getABD()
#         self.abd = np.linalg.inv(self.ABD)
#         self.getEngineeringConst()
#
#         self.load_angle = np.radians(load_angle)
#         # angle of load vector (degrees,
#         # converted to radians for internal maths)
#         self.m_load_angle = np.cos(self.load_angle)
#         self.n_load_angle = np.sin(self.load_angle)
#
#     def getABD(self):
#         # define the z-position of laminae. Datum = bottom
#         for i in range(self.n_plies):
#             self.z[i + 1] = (i + 1) * self.plies[i].t
#
#             # change datum of z-position from bottom to midplane
#         if self.midplane:
#             self.z -= np.max(self.z) / 2
#
#         for i in range(self.n_plies):
#             self.A += self.plies[i].Qbarmat * (self.z[i + 1] - self.z[i])
#             self.B += (
#                 1 / 2 * self.plies[i].Qbarmat * ((self.z[i + 1]) ** 2 - self.z[i] ** 2)
#             )
#             self.D += (
#                 1 / 3 * self.plies[i].Qbarmat * ((self.z[i + 1]) ** 3 - self.z[i] ** 3)
#             )
#         AB = np.concatenate((self.A, self.B), axis=0)
#         BD = np.concatenate((np.transpose(self.B), self.D), axis=0)
#         ABD = np.concatenate((AB, BD), axis=1)
#         return ABD
#
#     def getEngineeringConst(self):
#         self.Axx = self.A[0, 0]
#         self.Ayy = self.A[1, 1]
#         self.Ass = self.A[2, 2]
#         self.Axy = self.A[0, 1]
#
#         self.dxx = self.abd[3, 3]
#         self.dyy = self.abd[4, 4]
#         self.dss = self.abd[5, 5]
#         self.dxy = self.abd[3, 4]
#
#         self.A_const = self.Axx * self.Ayy - (self.Axy * self.Axy)
#         self.h = np.max(self.z) - np.min(self.z)
#
#         # in-plane engineering constants
#         self.Ex = self.A_const / (self.h * self.Ayy)
#         self.Ey = self.A_const / (self.h * self.Axx)
#         self.Gxy = self.Ass / self.h
#         self.vxy = self.Axy / self.Ayy
#         self.vyx = self.Axy / self.Axx
#
#         # flexural engineering constants
#         self.Exb = 12 / (self.h**3 * self.dxx)
#         self.Eyb = 12 / (self.h**3 * self.dyy)
#         self.Gxyb = 12 / (self.h**3 * self.dss)
#         self.vxyb = -self.dxy / self.dxx
#         self.vyxb = -self.dxy / self.dyy
#
#     def getStressStrain(self):
#         # compute global strain (laminate level) on mid-plane
#         self.load = np.array(
#             [self.Nx, self.Ny, self.Ns, self.Mx, self.My, self.Ms]
#         )  # .reshape(-1,1)
#
#         # self.strainMidplane = self.abd @ self.load # [e^0_x, e^0_y, e^0_s, kx, ky, ks]
#         self.strainMidplane = np.linalg.solve(
#             self.ABD.astype("float64"), self.load.astype("float64")
#         )
#         self.globalstrainVector = np.zeros((3 * self.n_plies))
#         self.globalstressVector = np.zeros((3 * self.n_plies))
#         self.localstrainVector = np.zeros((3 * self.n_plies))
#         self.localstressVector = np.zeros((3 * self.n_plies))
#         self.z_lamina_midplane = np.zeros(self.n_plies)
#
#         # compute local strain (lamina level)
#         for i in range(self.n_plies):
#             self.z_lamina_midplane[i] = 0.5 * (
#                 self.z[i + 1] + self.z[i]
#             )  # z-values from laminate datum (ie. laminate midplane) to laminae datums (ie. lamina midplane)
#
#             # global coordinate system (x,y)
#             self.globalstrainVector[3 * i : 3 * (i + 1)] = (
#                 self.strainMidplane[:3]
#                 + self.z_lamina_midplane[i] * self.strainMidplane[3:]
#             )  # [ex, ey, es], computed on the midplane of each lamina
#             self.globalstressVector[3 * i : 3 * (i + 1)] = (
#                 self.plies[i].Qbarmat @ self.globalstrainVector[3 * i : 3 * (i + 1)]
#             )
#
#             # local/principal coordinate system (1,2)
#             self.localstrainVector[3 * i : 3 * (i + 1)] = (
#                 self.plies[i].T @ self.globalstrainVector[3 * i : 3 * (i + 1)]
#             )
#             self.localstressVector[3 * i : 3 * (i + 1)] = (
#                 self.plies[i].T @ self.globalstressVector[3 * i : 3 * (i + 1)]
#             )
#
#         self.e11 = self.localstrainVector[0::3]
#         self.e22 = self.localstrainVector[1::3]
#         self.e12 = self.localstrainVector[2::3]
#
#         self.sigma11 = self.localstressVector[0::3]
#         self.sigma22 = self.localstressVector[1::3]
#         self.sigma12 = self.localstressVector[2::3]
#
#     def getstressstrainEnvelope(self):
#         # compute global strain (laminate level) on mid-plane
#         self.load = np.array([self.Nx, self.Ny, self.Ns, self.Mx, self.My, self.Ms])
#         self.sigmax = self.Nx / self.h
#         self.sigmay = (self.Ny / self.h) * np.cos(self.load_angle)
#         self.sigmas = (self.Ns / self.h) * np.sin(self.load_angle)
#         self.sigma = np.array([self.sigmax, self.sigmay, self.sigmas]).reshape(-1, 1)
#
#         self.Sbarmat = np.linalg.inv(self.A) * self.h
#
#         self.sigmaprime = self.sigma
#         self.sigmaxprime = self.sigmaprime[0]
#         self.sigmayprime = self.sigmaprime[1]
#         self.sigmaxyprime = self.sigmaprime[2]
#
#         self.strainMidplane = self.Sbarmat @ self.sigmaprime
#
#         self.globalstrainVector = np.zeros((3 * self.n_plies))
#         self.globalstressVector = np.zeros((3 * self.n_plies))
#         self.localstrainVector = np.zeros((3 * self.n_plies))
#         self.localstressVector = np.zeros((3 * self.n_plies))
#         self.z_lamina_midplane = np.zeros(self.n_plies)
#
#         # compute local strain (lamina level)
#         for i in range(self.n_plies):
#             self.z_lamina_midplane[i] = 0.5 * (
#                 self.z[i + 1] + self.z[i]
#             )  # z-values from laminate datum (ie: laminate midplane) to laminas' datums (ie: lamina midplane)
#
#             # global coordinate system (x,y)
#             self.globalstrainVector[3 * i : 3 * (i + 1)] = np.ravel(
#                 self.strainMidplane[:3]
#             )  # [ex, ey, es], computed on the midplane of each lamina
#             self.globalstressVector[3 * i : 3 * (i + 1)] = (
#                 self.plies[i].Qbarmat @ self.globalstrainVector[3 * i : 3 * (i + 1)]
#             )
#
#             # local/principal coordinate system (1,2)
#             self.localstrainVector[3 * i : 3 * (i + 1)] = (
#                 self.plies[i].T @ self.globalstrainVector[3 * i : 3 * (i + 1)]
#             )
#             self.localstressVector[3 * i : 3 * (i + 1)] = (
#                 self.plies[i].T @ self.globalstressVector[3 * i : 3 * (i + 1)]
#             )
#
#         self.e11 = self.localstrainVector[0::3]
#         self.e22 = self.localstrainVector[1::3]
#         self.e12 = self.localstrainVector[2::3]
#         self.exglobal = self.strainMidplane[0]
#         self.eyglobal = self.strainMidplane[1]
#         self.esglobal = self.strainMidplane[2]
#
#         self.sigma11 = self.localstressVector[0::3]
#         self.sigma22 = self.localstressVector[1::3]
#         self.sigma12 = self.localstressVector[2::3]
