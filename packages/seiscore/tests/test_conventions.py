import numpy as np

from seiscore.conventions import canonical_component, ray_frames


def test_frames_are_orthonormal_and_match_definitions():
    src = np.array([10.0, -20.0, -2500.0])
    rec = np.array([[500.0, 300.0, 0.0], [-800.0, 50.0, 10.0]])
    frames, valid = ray_frames(src, rec)
    assert valid.all()
    for f in frames:
        np.testing.assert_allclose(f @ f.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.cross(f[1], f[0]), f[2], atol=1e-12)  # SV = SH x P
        np.testing.assert_allclose(
            np.cross([0, 0, 1], f[0])[:2] / np.linalg.norm(f[0][:2]), f[1][:2], atol=1e-12
        )  # SH = unit(z x P)


def test_p_points_from_source_to_receiver():
    frames, _ = ray_frames(np.zeros(3), np.array([[0.0, 1000.0, 1000.0]]))
    p, sh, sv = frames[0]
    np.testing.assert_allclose(p, [0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    np.testing.assert_allclose(sh, [-1, 0, 0])  # z x P
    assert sh[2] == 0.0
    assert sv[2] < 0  # SV = SH x P points down-ish for an up-going ray toward +y


def test_vertical_and_bad_rays_are_invalid():
    rec = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 0.0, -100.0]])
    frames, valid = ray_frames(np.array([0.0, 0.0, -100.0]), rec)
    assert not valid.any()
    assert np.all(frames == 0)


def test_component_aliases():
    assert canonical_component("East") == "E"
    assert canonical_component("vertical") == "Z"
    assert canonical_component("P") is None
