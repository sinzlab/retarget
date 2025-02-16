import operator

import numpy as np
import copy

import retarget.utils.AnimationStructure as AnimationStructure
from retarget.model import graph_to_batch, mask_from_batch
from retarget.utils.Quaternions_old import Quaternions, d6_2_rotmat


class Animation:
    """
    Animation is a numpy-like wrapper for animation data

    Animation data consists of several arrays consisting
    of F frames and J joints.

    The animation is specified by

        rotations : (F, J) Quaternions | Joint Rotations
        positions : (F, J, 3) ndarray  | Joint Positions

    The base pose is specified by

        orients   : (J) Quaternions    | Joint Orientations
        offsets   : (J, 3) ndarray     | Joint Offsets

    And the skeletal structure is specified by

        parents   : (J) ndarray        | Joint Parents
    """

    def __init__(self, rotations, positions, orients, offsets, parents):

        self.rotations = rotations
        self.positions = positions
        self.orients = orients
        self.offsets = offsets
        self.parents = parents

        self.positions = forward_rotations(
            self.parents, self.offsets, self.rotations, self.positions[:, 0]
        )

        self.edges = AnimationStructure.edges(self.parents)
        self.t_pose = AnimationStructure.t_pose(self.offsets, self.edges)

    def __op__(self, op, other):
        return Animation(
            op(self.rotations, other.rotations),
            op(self.positions, other.positions),
            op(self.orients, other.orients),
            op(self.offsets, other.offsets),
            op(self.parents, other.parents),
        )

    def __iop__(self, op, other):
        self.rotations = op(self.roations, other.rotations)
        self.positions = op(self.roations, other.positions)
        self.orients = op(self.orients, other.orients)
        self.offsets = op(self.offsets, other.offsets)
        self.parents = op(self.parents, other.parents)
        return self

    def __sop__(self, op):
        return Animation(
            op(self.rotations),
            op(self.positions),
            op(self.orients),
            op(self.offsets),
            op(self.parents),
        )

    def __add__(self, other):
        return self.__op__(operator.add, other)

    def __sub__(self, other):
        return self.__op__(operator.sub, other)

    def __mul__(self, other):
        return self.__op__(operator.mul, other)

    def __div__(self, other):
        return self.__op__(operator.div, other)

    def __abs__(self):
        return self.__sop__(operator.abs)

    def __neg__(self):
        return self.__sop__(operator.neg)

    def __iadd__(self, other):
        return self.__iop__(operator.iadd, other)

    def __isub__(self, other):
        return self.__iop__(operator.isub, other)

    def __imul__(self, other):
        return self.__iop__(operator.imul, other)

    def __idiv__(self, other):
        return self.__iop__(operator.idiv, other)

    def __len__(self):
        return len(self.rotations)

    def __getitem__(self, k):
        if isinstance(k, tuple):
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients[k[1:]],
                self.offsets[k[1:]],
                self.parents[k[1:]],
            )
        else:
            return Animation(
                self.rotations[k],
                self.positions[k],
                self.orients,
                self.offsets,
                self.parents,
            )

    def __setitem__(self, k, v):
        if isinstance(k, tuple):
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k[1:], v.orients)
            self.offsets.__setitem__(k[1:], v.offsets)
            self.parents.__setitem__(k[1:], v.parents)
        else:
            self.rotations.__setitem__(k, v.rotations)
            self.positions.__setitem__(k, v.positions)
            self.orients.__setitem__(k, v.orients)
            self.offsets.__setitem__(k, v.offsets)
            self.parents.__setitem__(k, v.parents)

    @property
    def shape(self):
        return (self.rotations.shape[0], self.rotations.shape[1])

    def copy(self):
        return Animation(
            self.rotations.copy(),
            self.positions.copy(),
            self.orients.copy(),
            self.offsets.copy(),
            self.parents.copy(),
        )

    def repeat(self, *args, **kw):
        return Animation(
            self.rotations.repeat(*args, **kw),
            self.positions.repeat(*args, **kw),
            self.orients,
            self.offsets,
            self.parents,
        )

    def ravel(self):
        return np.hstack(
            [
                self.rotations.log().ravel(),
                self.positions.ravel(),
                self.orients.log().ravel(),
                self.offsets.ravel(),
            ]
        )

    @classmethod
    def unravel(clas, anim, shape, parents):
        nf, nj = shape
        rotations = anim[nf * nj * 0 : nf * nj * 3]
        positions = anim[nf * nj * 3 : nf * nj * 6]
        orients = anim[nf * nj * 6 + nj * 0 : nf * nj * 6 + nj * 3]
        offsets = anim[nf * nj * 6 + nj * 3 : nf * nj * 6 + nj * 6]
        return cls(
            Quaternions.exp(rotations),
            positions,
            Quaternions.exp(orients),
            offsets,
            parents.copy(),
        )

    def as_graph(self, stride=1):
        """
        Convert Animation to Graph Data.

        Such that you can run

        ```python
        animation, _, _ = load('motion.bvh')
        graph = animation.as_graph()
        ```

        Parameters
        ----------
        animation : Animation
            Input animation

        stride : int
            Stride to sample the animation

        Returns
        -------
        data : [Data]
            List of Graph Data objects

        """
        from torch_geometric.data import Data
        from retarget.utils.Quaternions_old import quat_2_d6

        data = []

        for i,(position, rotation) in enumerate(zip(
                self.positions[::stride], self.rotations[::stride]
            )):
            d6 = quat_2_d6(rotation)
                
            position = torch.Tensor(position)
            root_trajectory = torch.zeros_like(position)
            root_trajectory[0] += position[0]
            position = position - position[0]

            if i * stride == 0:
                position_prev = torch.Tensor(position)
                rotation_prev = torch.Tensor(rotation)
            else:
                idx_prev = (i-1)*stride
                position_prev = torch.Tensor(self.positions[idx_prev])
                rotation_prev = torch.Tensor(np.array(self.rotations[idx_prev]))

            position_prev = position_prev - position_prev[0]

            velocity = position - position_prev
                
            rotation = torch.Tensor(rotation)
            d6 = torch.Tensor(d6)

            t_pose = torch.Tensor(self.t_pose)
            offsets = torch.Tensor(self.offsets)
            parents = torch.LongTensor(self.parents)
            edges = torch.LongTensor(self.edges.T)


            features = torch.cat([d6, position, position_prev, velocity, root_trajectory], dim=-1)

            data.append(
                Data(
                    features,
                    edges,
                    rotation=rotation,
                    position=position,
                    d6=d6,
                    root_trajectory = root_trajectory,
                    rotation_prev = rotation_prev,
                    pos=t_pose,
                    offsets=offsets,
                    parents=parents,
                )
            )

        return data


""" Maya Interaction """


def load_to_maya(anim, names=None, radius=0.5):
    """
    Load Animation Object into Maya as Joint Skeleton
    loads each frame as a new keyfame in maya.

    If the animation is too slow or too fast perhaps
    the framerate needs adjusting before being loaded
    such that it matches the maya scene framerate.


    Parameters
    ----------

    anim : Animation
        Animation to load into Scene

    names : [str]
        Optional list of Joint names for Skeleton

    Returns
    -------

    List of Maya Joint Nodes loaded into scene
    """

    import pymel.core as pm

    joints = []
    frames = range(1, len(anim) + 1)

    if names is None:
        names = ["joint_" + str(i) for i in range(len(anim.parents))]

    for i, offset, orient, parent, name in zip(
        range(len(anim.offsets)), anim.offsets, anim.orients, anim.parents, names
    ):

        if parent < 0:
            pm.select(d=True)
        else:
            pm.select(joints[parent])

        joint = pm.joint(n=name, p=offset, relative=True, radius=radius)
        joint.setOrientation([orient[1], orient[2], orient[3], orient[0]])

        curvex = pm.nodetypes.AnimCurveTA(n=name + "_rotateX")
        curvey = pm.nodetypes.AnimCurveTA(n=name + "_rotateY")
        curvez = pm.nodetypes.AnimCurveTA(n=name + "_rotateZ")

        jrotations = (-Quaternions(orient[np.newaxis]) * anim.rotations[:, i]).euler()
        curvex.addKeys(frames, jrotations[:, 0])
        curvey.addKeys(frames, jrotations[:, 1])
        curvez.addKeys(frames, jrotations[:, 2])

        pm.connectAttr(curvex.output, joint.rotateX)
        pm.connectAttr(curvey.output, joint.rotateY)
        pm.connectAttr(curvez.output, joint.rotateZ)

        offsetx = pm.nodetypes.AnimCurveTU(n=name + "_translateX")
        offsety = pm.nodetypes.AnimCurveTU(n=name + "_translateY")
        offsetz = pm.nodetypes.AnimCurveTU(n=name + "_translateZ")

        offsetx.addKeys(frames, anim.positions[:, i, 0])
        offsety.addKeys(frames, anim.positions[:, i, 1])
        offsetz.addKeys(frames, anim.positions[:, i, 2])

        pm.connectAttr(offsetx.output, joint.translateX)
        pm.connectAttr(offsety.output, joint.translateY)
        pm.connectAttr(offsetz.output, joint.translateZ)

        joints.append(joint)

    return joints


def load_from_maya(root, start, end):
    """
    Load Animation Object from Maya Joint Skeleton

    Parameters
    ----------

    root : PyNode
        Root Joint of Maya Skeleton

    start, end : int, int
        Start and End frame index of Maya Animation

    Returns
    -------

    animation : Animation
        Loaded animation from maya

    names : [str]
        Joint names from maya
    """

    import pymel.core as pm

    original_time = pm.currentTime(q=True)
    pm.currentTime(start)

    """ Build Structure """

    names, parents = AnimationStructure.load_from_maya(root)
    descendants = AnimationStructure.descendants_list(parents)
    orients = Quaternions.id(len(names))
    offsets = np.array([pm.xform(j, q=True, translation=True) for j in names])

    for j, name in enumerate(names):
        scale = pm.xform(pm.PyNode(name), q=True, scale=True, relative=True)
        if len(descendants[j]) == 0:
            continue
        offsets[descendants[j]] *= scale

    """ Load Animation """

    eulers = np.zeros((end - start, len(names), 3))
    positions = np.zeros((end - start, len(names), 3))
    rotations = Quaternions.id((end - start, len(names)))

    for i in range(end - start):

        pm.currentTime(start + i + 1, u=True)

        scales = {}

        for j, name, parent in zip(range(len(names)), names, parents):

            node = pm.PyNode(name)

            if i == 0 and pm.hasAttr(node, "jointOrient"):
                ort = node.getOrientation()
                orients[j] = Quaternions(np.array([ort[3], ort[0], ort[1], ort[2]]))

            if pm.hasAttr(node, "rotate"):
                eulers[i, j] = np.radians(pm.xform(node, q=True, rotation=True))
            if pm.hasAttr(node, "translate"):
                positions[i, j] = pm.xform(node, q=True, translation=True)
            if pm.hasAttr(node, "scale"):
                scales[j] = pm.xform(node, q=True, scale=True, relative=True)

        for j in scales:
            if len(descendants[j]) == 0:
                continue
            positions[i, descendants[j]] *= scales[j]

        positions[i, 0] = pm.xform(root, q=True, translation=True, worldSpace=True)

    rotations = orients[np.newaxis] * Quaternions.from_euler(
        eulers, order="xyz", world=True
    )

    """ Done """

    pm.currentTime(original_time)

    return Animation(rotations, positions, orients, offsets, parents), names


# local transformation matrices
def transforms_local(anim):
    """
    Computes Animation Local Transforms

    As well as a number of other uses this can
    be used to compute global joint transforms,
    which in turn can be used to compete global
    joint positions

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray

        For each frame F, joint local
        transforms for each joint J
    """

    transforms = anim.rotations.transforms()
    transforms = np.concatenate(
        [transforms, np.zeros(transforms.shape[:2] + (3, 1))], axis=-1
    )
    transforms = np.concatenate(
        [transforms, np.zeros(transforms.shape[:2] + (1, 4))], axis=-2
    )
    # the last column is filled with the joint positions!
    transforms[:, :, 0:3, 3] = anim.positions
    transforms[:, :, 3:4, 3] = 1.0
    return transforms


def transforms_multiply(t0s, t1s):
    """
    Transforms Multiply

    Multiplies two arrays of animation transforms

    Parameters
    ----------

    t0s, t1s : (F, J, 4, 4) ndarray
        Two arrays of transforms
        for each frame F and each
        joint J

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray
        Array of transforms for each
        frame F and joint J multiplied
        together
    """

    return t0s @ t1s


def transforms_inv(ts):
    fts = ts.reshape(-1, 4, 4)
    fts = np.array(list(map(lambda x: np.linalg.inv(x), fts)))
    return fts.reshape(ts.shape)


def transforms_blank(anim):
    """
    Blank Transforms

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    transforms : (F, J, 4, 4) ndarray
        Array of identity transforms for
        each frame F and joint J
    """

    ts = np.zeros(anim.shape + (4, 4))
    ts[:, :, 0, 0] = 1.0
    ts[:, :, 1, 1] = 1.0
    ts[:, :, 2, 2] = 1.0
    ts[:, :, 3, 3] = 1.0
    return ts


# global transformation matrices
def transforms_global(anim):
    """
    Global Animation Transforms

    This relies on joint ordering
    being incremental. That means a joint
    J1 must not be a ancestor of J0 if
    J0 appears before J1 in the joint
    ordering.

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    ------

    transforms : (F, J, 4, 4) ndarray
        Array of global transforms for
        each frame F and joint J
    """

    joints = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals = transforms_local(anim)
    globals = transforms_blank(anim)

    globals[:, 0] = locals[:, 0]

    for i in range(1, anim.shape[1]):
        globals[:, i] = transforms_multiply(globals[:, anim.parents[i]], locals[:, i])

    return globals


# !!! useful!
def positions_global(anim):
    """
    Global Joint Positions

    Given an animation compute the global joint
    positions at at every frame

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    positions : (F, J, 3) ndarray
        Positions for every frame F
        and joint position J
    """

    # get the last column -- corresponding to the coordinates
    positions = transforms_global(anim)[:, :, :, 3]
    return positions[:, :, :3] / positions[:, :, 3, np.newaxis]


""" Rotations """


def rotations_global(anim):
    """
    Global Animation Rotations

    This relies on joint ordering
    being incremental. That means a joint
    J1 must not be a ancestor of J0 if
    J0 appears before J1 in the joint
    ordering.

    Parameters
    ----------

    anim : Animation
        Input animation

    Returns
    -------

    points : (F, J) Quaternions
        global rotations for every frame F
        and joint J
    """

    joints = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals = anim.rotations
    globals = Quaternions.id(anim.shape)

    globals[:, 0] = locals[:, 0]

    for i in range(1, anim.shape[1]):
        globals[:, i] = globals[:, anim.parents[i]] * locals[:, i]

    return globals


def rotations_parents_global(anim):
    rotations = rotations_global(anim)
    rotations = rotations[:, anim.parents]
    rotations[:, 0] = Quaternions.id(len(anim))
    return rotations


def rotations_load_to_maya(rotations, positions, names=None):
    """
    Load Rotations into Maya

    Loads a Quaternions array into the scene
    via the representation of axis

    Parameters
    ----------

    rotations : (F, J) Quaternions
        array of rotations to load
        into the scene where
            F = number of frames
            J = number of joints

    positions : (F, J, 3) ndarray
        array of positions to load
        rotation axis at where:
            F = number of frames
            J = number of joints

    names : [str]
        List of joint names

    Returns
    -------

    maxies : Group
        Grouped Maya Node of all Axis nodes
    """

    import pymel.core as pm

    if names is None:
        names = ["joint_" + str(i) for i in range(rotations.shape[1])]

    maxis = []
    frames = range(1, len(positions) + 1)
    for i, name in enumerate(names):

        name = name + "_axis"
        axis = pm.group(
            pm.curve(p=[(0, 0, 0), (1, 0, 0)], d=1, n=name + "_axis_x"),
            pm.curve(p=[(0, 0, 0), (0, 1, 0)], d=1, n=name + "_axis_y"),
            pm.curve(p=[(0, 0, 0), (0, 0, 1)], d=1, n=name + "_axis_z"),
            n=name,
        )

        axis.rotatePivot.set((0, 0, 0))
        axis.scalePivot.set((0, 0, 0))
        axis.childAtIndex(0).overrideEnabled.set(1)
        axis.childAtIndex(0).overrideColor.set(13)
        axis.childAtIndex(1).overrideEnabled.set(1)
        axis.childAtIndex(1).overrideColor.set(14)
        axis.childAtIndex(2).overrideEnabled.set(1)
        axis.childAtIndex(2).overrideColor.set(15)

        curvex = pm.nodetypes.AnimCurveTA(n=name + "_rotateX")
        curvey = pm.nodetypes.AnimCurveTA(n=name + "_rotateY")
        curvez = pm.nodetypes.AnimCurveTA(n=name + "_rotateZ")

        arotations = rotations[:, i].euler()
        curvex.addKeys(frames, arotations[:, 0])
        curvey.addKeys(frames, arotations[:, 1])
        curvez.addKeys(frames, arotations[:, 2])

        pm.connectAttr(curvex.output, axis.rotateX)
        pm.connectAttr(curvey.output, axis.rotateY)
        pm.connectAttr(curvez.output, axis.rotateZ)

        offsetx = pm.nodetypes.AnimCurveTU(n=name + "_translateX")
        offsety = pm.nodetypes.AnimCurveTU(n=name + "_translateY")
        offsetz = pm.nodetypes.AnimCurveTU(n=name + "_translateZ")

        offsetx.addKeys(frames, positions[:, i, 0])
        offsety.addKeys(frames, positions[:, i, 1])
        offsetz.addKeys(frames, positions[:, i, 2])

        pm.connectAttr(offsetx.output, axis.translateX)
        pm.connectAttr(offsety.output, axis.translateY)
        pm.connectAttr(offsetz.output, axis.translateZ)

        maxis.append(axis)

    return pm.group(*maxis, n="RotationAnimation")


def forward_rotations(parents, offset, rotations, trajectory=None):
    """
    input: rotations [T, J, 4], rtpos [T, 3]
    output: positions [T, J, 3]
    """
    transforms = rotations.transforms()  # [..., J, 3, 3]
    glb = np.zeros(rotations.shape + (3,))  # [T, J, 3]

    if trajectory is not None:
        glb[..., 0, :] = trajectory

    for i, pi in enumerate(parents):
        if pi == -1:
            continue

        glb[..., i, :] = np.matmul(transforms[..., pi, :, :], offset[i])
        glb[..., i, :] += glb[..., pi, :]
        transforms[..., i, :, :] = np.matmul(
            transforms[..., pi, :, :], transforms[..., i, :, :]
        )
    return glb


import torch


def forward_rotations_torch(edges, offset, rotations, trajectory=None, device="cpu"):
    """
    input: rotations [T, J, 4], rtpos [T, 3]
    output: positions [T, J, 3]
    """

    rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)

    transforms = transform_from_quaternion(rotations)  # [..., J, 3, 3]
    result = torch.zeros(rotations.shape[:-1] + (3,), device=device)

    topology = [-1] * (len(edges) + 1)
    for i, edge in enumerate(edges):
        topology[edge[1]] = edge[0].item()

    if trajectory is not None:
        result[..., 0, :] = trajectory

    for i, pi in enumerate(topology):
        if pi == -1:
            continue

        result[..., i, :] = torch.matmul(
            transforms[..., pi, :, :], offset[i]
        )  # .squeeze()
        result[..., i, :] += result[..., pi, :]

        transforms[..., i, :, :] = torch.matmul(
            transforms[..., pi, :, :].clone(), transforms[..., i, :, :].clone()
        )

    return result


def forward_rotations_torch_batch(
    edge_indexs, offset, rotations, trajectory=None, quater=True, device="cuda"
):
    """
    Assumes a padded batched input, such that the edge_indexes are padded with -1, offset is padded with 0 and rotations are padded with 1.
    input: edge_indexs [B, J, 2], offset [B, J, 3], rotations [B, J, 4]
    output: positions [B, J, 3]
    """

    if quater:
        rotations = rotations / torch.norm(rotations, dim=-1, keepdim=True)

        transforms = transform_from_quaternion(rotations)  # [..., J, 3, 3]
    else:
        transforms = rotations

    result = torch.zeros(rotations.shape[0], rotations.shape[1] + 1, 3, device=device)

    transforms = torch.cat(
        [
            transforms,
            torch.eye(3)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(transforms.shape[0], transforms.shape[1], 1, 1)
            .to(device),
        ],
        dim=1,
    )
    offset = torch.cat([offset, torch.zeros_like(offset[:, :1, :])], dim=1).to(device)

    topology = -torch.ones(
        edge_indexs.shape[0], edge_indexs.shape[1] + 2, dtype=torch.long, device=device
    )
    for i in range(edge_indexs.shape[1]):
        topology[np.arange(len(edge_indexs)), edge_indexs[:, i, 1]] = edge_indexs[
            :, i, 0
        ]

    topology = topology[..., :-1]

    if trajectory is not None:
        result[..., 0, :] = trajectory

    for i, pi in enumerate(topology.T):
        ci = torch.ones_like(pi) * i

        result[np.arange(len(pi)), ci, :] = torch.matmul(
            transforms[np.arange(len(pi)), pi, :, :],
            offset[np.arange(len(pi)), ci, None].transpose(1, 2),
        ).squeeze()

        result[np.arange(len(pi)), ci, :] += result[np.arange(len(pi)), pi, :]

        transforms[np.arange(len(pi)), ci, :, :] = torch.matmul(
            transforms[np.arange(len(pi)), pi, :, :].clone(),
            transforms[np.arange(len(pi)), ci, :, :].clone(),
        )

    result = result[:, :-1]
    return result


def fk_for_batch(
    batch, replace_rotations=None, quater=True, device="cuda", rotations_fmt="rotmat"
):
    if rotations_fmt == "d6":
        replace_rotations = d6_2_rotmat(replace_rotations)

    mask = mask_from_batch(batch)
    offsets = graph_to_batch(batch.offsets, mask, pad_with=0)
    rotations_poses = graph_to_batch(batch.x, mask, pad_with=1)
    rotations = (
        replace_rotations
        if replace_rotations is not None
        else rotations_poses[..., :-3]
    )

    max_length = mask.sum(1).max().item() - 1

    data_list = batch.to_data_list()

    edge_indexs = -torch.ones(
        mask.shape[0], max_length, 2, dtype=torch.long, device=device
    )
    for i, item in enumerate(data_list):
        edge_indexs[i, : len(item.edge_index.T)] = item.edge_index.T

    poses = forward_rotations_torch_batch(
        edge_indexs, offsets, rotations, device=device, quater=quater
    )

    return poses, edge_indexs


def transform_from_quaternion(quater: torch.Tensor):
    qw = quater[..., 0]
    qx = quater[..., 1]
    qy = quater[..., 2]
    qz = quater[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


""" Offsets & Orients """


def orients_global(anim):

    joints = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals = anim.orients
    globals = Quaternions.id(anim.shape[1])

    globals[:, 0] = locals[:, 0]

    for i in range(1, anim.shape[1]):
        globals[:, i] = globals[:, anim.parents[i]] * locals[:, i]

    return globals


def offsets_transforms_local(anim):

    transforms = anim.orients[np.newaxis].transforms()
    transforms = np.concatenate(
        [transforms, np.zeros(transforms.shape[:2] + (3, 1))], axis=-1
    )
    transforms = np.concatenate(
        [transforms, np.zeros(transforms.shape[:2] + (1, 4))], axis=-2
    )
    transforms[:, :, 0:3, 3] = anim.offsets[np.newaxis]
    transforms[:, :, 3:4, 3] = 1.0
    return transforms


def offsets_transforms_global(anim):

    joints = np.arange(anim.shape[1])
    parents = np.arange(anim.shape[1])
    locals = offsets_transforms_local(anim)
    globals = transforms_blank(anim)

    globals[:, 0] = locals[:, 0]

    for i in range(1, anim.shape[1]):
        globals[:, i] = transforms_multiply(globals[:, anim.parents[i]], locals[:, i])

    return globals


def offsets_global(anim):
    offsets = offsets_transforms_global(anim)[:, :, :, 3]
    return offsets[0, :, :3] / offsets[0, :, 3, np.newaxis]


""" Lengths """


def offset_lengths(anim):
    return np.sum(anim.offsets[1:] ** 2.0, axis=1) ** 0.5


def position_lengths(anim):
    return np.sum(anim.positions[:, 1:] ** 2.0, axis=2) ** 0.5


""" Skinning """


def skin(anim, rest, weights, mesh, maxjoints=4):

    full_transforms = transforms_multiply(
        transforms_global(anim), transforms_inv(transforms_global(rest[0:1]))
    )

    weightids = np.argsort(-weights, axis=1)[:, :maxjoints]
    weightvls = np.array(list(map(lambda w, i: w[i], weights, weightids)))
    weightvls = weightvls / weightvls.sum(axis=1)[..., np.newaxis]

    verts = np.hstack([mesh, np.ones((len(mesh), 1))])
    verts = verts[np.newaxis, :, np.newaxis, :, np.newaxis]
    verts = transforms_multiply(full_transforms[:, weightids], verts)
    verts = (verts[:, :, :, :3] / verts[:, :, :, 3:4])[:, :, :, :, 0]

    return np.sum(weightvls[np.newaxis, :, :, np.newaxis] * verts, axis=2)
