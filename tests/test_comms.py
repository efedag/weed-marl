import numpy as np
import pytest

from sim.comms import (
    BernoulliChannel,
    GilbertElliottChannel,
    Message,
    PerfectChannel,
    make_channel,
)
from sim.weed_env import STAY, WeedEnv, WeedEnvConfig

N_LONG = 200_000


def single_link_losses(channel, n, seed=0, sender="drone_0", receiver="robot_0"):
    """Tek bağlantıda art arda n mesaj; her adımda bir mesaj gönderilir."""
    channel.reset(np.random.default_rng(seed))
    lost = np.empty(n, dtype=bool)
    batch = [Message(sender, receiver, 0)]
    for i in range(n):
        lost[i] = not channel.transmit(batch)
    return lost


def bulk_losses(channel, n, seed=0):
    """Tek bağlantıda n mesajı tek çağrıda gönderir (sıra korunur)."""
    channel.reset(np.random.default_rng(seed))
    messages = [Message("drone_0", "robot_0", i, i) for i in range(n)]
    delivered = {m.payload for m in channel.transmit(messages)}
    return np.array([i not in delivered for i in range(n)])


def mean_burst_length(lost):
    """Ardışık kayıp dizilerinin ortalama uzunluğu."""
    padded = np.concatenate([[False], lost, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return (ends - starts).mean()


def test_perfect_channel_never_loses():
    lost = bulk_losses(PerfectChannel(), 1000)
    assert not lost.any()


def test_bernoulli_p0_loses_nothing_and_p1_loses_everything():
    assert not bulk_losses(BernoulliChannel(0.0), 5000).any()
    assert bulk_losses(BernoulliChannel(1.0), 5000).all()


def test_gilbert_elliott_extremes():
    assert not bulk_losses(GilbertElliottChannel.from_mean_loss(0.0), 5000).any()
    assert bulk_losses(GilbertElliottChannel.from_mean_loss(1.0), 5000).all()


def test_counters_track_sent_and_lost():
    channel = BernoulliChannel(0.5)
    lost = bulk_losses(channel, 10_000)
    assert channel.sent == 10_000
    assert channel.lost == int(lost.sum())
    assert channel.loss_rate == pytest.approx(lost.mean())
    channel.reset(np.random.default_rng(1))
    assert channel.sent == channel.lost == 0


@pytest.mark.parametrize("p", [0.1, 0.25, 0.5])
def test_bernoulli_empirical_loss_rate(p):
    lost = bulk_losses(BernoulliChannel(p), N_LONG, seed=11)
    assert abs(lost.mean() - p) <= 0.01


@pytest.mark.parametrize("mean_loss", [0.25, 0.5])
def test_gilbert_elliott_empirical_loss_rate(mean_loss):
    channel = GilbertElliottChannel.from_mean_loss(mean_loss, mean_burst_length=5)
    assert channel.mean_loss == pytest.approx(mean_loss)
    lost = bulk_losses(channel, N_LONG, seed=12)
    assert abs(lost.mean() - mean_loss) <= 0.01


@pytest.mark.parametrize("mean_loss", [0.25, 0.5])
def test_gilbert_elliott_bursts_are_longer_than_bernoulli(mean_loss):
    bernoulli = bulk_losses(BernoulliChannel(mean_loss), N_LONG, seed=13)
    ge = bulk_losses(
        GilbertElliottChannel.from_mean_loss(mean_loss, mean_burst_length=5), N_LONG, seed=13
    )
    bern_burst, ge_burst = mean_burst_length(bernoulli), mean_burst_length(ge)
    # Bernoulli için beklenen 1/(1-p): 1.33 ve 2.0; G-E için ~5
    assert bern_burst == pytest.approx(1 / (1 - mean_loss), rel=0.05)
    assert ge_burst == pytest.approx(5.0, rel=0.1)
    assert ge_burst > 2 * bern_burst


def test_step_by_step_and_bulk_transmission_agree():
    ge = GilbertElliottChannel.from_mean_loss(0.25, mean_burst_length=4)
    one_by_one = single_link_losses(ge, 20_000, seed=5)
    assert abs(one_by_one.mean() - 0.25) <= 0.02
    assert mean_burst_length(one_by_one) == pytest.approx(4.0, rel=0.15)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BernoulliChannel(0.3),
        lambda: GilbertElliottChannel.from_mean_loss(0.3, mean_burst_length=3),
    ],
)
def test_same_seed_same_loss_sequence(factory):
    a = bulk_losses(factory(), 5000, seed=21)
    b = bulk_losses(factory(), 5000, seed=21)
    c = bulk_losses(factory(), 5000, seed=22)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gilbert_elliott_keeps_state_per_link():
    channel = GilbertElliottChannel(p_gb=0.0, p_bg=0.0 + 1e-12, e_g=0.0, e_b=1.0)
    channel.reset(np.random.default_rng(0))
    channel._links[("drone_0", "robot_0")] = 1  # Bad
    channel._links[("drone_0", "robot_1")] = 0  # Good
    delivered = channel.transmit(
        [Message("drone_0", "robot_0", 0), Message("drone_0", "robot_1", 0)]
    )
    assert [m.receiver for m in delivered] == ["robot_1"]
    assert channel.link_state("drone_0", "robot_0") == 1
    assert channel.link_state("drone_0", "robot_1") == 0


def test_from_mean_loss_parameters():
    channel = GilbertElliottChannel.from_mean_loss(0.25, mean_burst_length=5, e_g=0.0, e_b=1.0)
    assert channel.p_bg == pytest.approx(0.2)
    assert channel.stationary_bad == pytest.approx(0.25)
    assert channel.p_gb == pytest.approx(0.25 * 0.2 / 0.75)
    assert channel.mean_burst_length == pytest.approx(5.0)
    with pytest.raises(ValueError):
        GilbertElliottChannel.from_mean_loss(0.9, mean_burst_length=1.0)
    with pytest.raises(ValueError):
        GilbertElliottChannel.from_mean_loss(0.5, mean_burst_length=0.5)


def test_make_channel():
    assert isinstance(make_channel("perfect"), PerfectChannel)
    assert make_channel("bernoulli", {"p": 0.1}).p == 0.1
    ge = make_channel("gilbert_elliott", {"mean_loss": 0.5, "mean_burst_length": 4})
    assert ge.mean_loss == pytest.approx(0.5)
    raw = make_channel("gilbert_elliott", {"p_gb": 0.1, "p_bg": 0.3})
    assert raw.p_gb == 0.1
    with pytest.raises(ValueError):
        make_channel("udp")


# ---------------------------------------------------------------- ortam entegrasyonu
def run_with_channel(channel, params, seed):
    config = WeedEnvConfig(
        grid_size=32,
        weed_density=0.2,
        n_drones=2,
        n_robots=2,
        max_steps=60,
        channel=channel,
        channel_params=params,
    )
    env = WeedEnv(config)
    env.reset(seed=seed)
    moves = np.random.default_rng(seed).integers(1, 5, size=(config.max_steps, 2))
    infos = {}
    known_per_step = []
    for t in range(config.max_steps):
        actions = {"drone_0": int(moves[t, 0]), "drone_1": int(moves[t, 1])}
        actions.update({robot: STAY for robot in env.robots})
        _, _, _, _, infos = env.step(actions)
        known_per_step.append(int(env.robot_known_weed.sum()))
    # Bölüm boyunca robotların bildiği ot sayısının ortalaması. Ardışık raporlar
    # örtüştüğü için bölüm sonundaki değer doyuma ulaşabilir; ortalama, bilginin
    # ne kadar geç geldiğini de yansıtır.
    return float(np.mean(known_per_step)), infos["robot_0"]["episode"], env


def test_channel_does_not_change_field_or_drone_paths():
    _, _, perfect = run_with_channel("perfect", {}, seed=3)
    _, _, lossy = run_with_channel("bernoulli", {"p": 0.5}, seed=3)
    np.testing.assert_array_equal(perfect.field, lossy.field)
    np.testing.assert_array_equal(perfect.drone_explored, lossy.drone_explored)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_more_loss_means_robots_know_fewer_weeds(seed):
    known = {}
    for name, params in [
        ("perfect", {}),
        ("bernoulli", {"p": 0.25}),
        ("bernoulli", {"p": 0.5}),
        ("gilbert_elliott", {"mean_loss": 0.5, "mean_burst_length": 5}),
        ("bernoulli", {"p": 1.0}),
    ]:
        known[(name, params.get("p", params.get("mean_loss")))], _, _ = run_with_channel(
            name, params, seed
        )
    assert known[("perfect", None)] > known[("bernoulli", 0.25)] > known[("bernoulli", 0.5)]
    assert known[("bernoulli", 0.5)] > known[("bernoulli", 1.0)] == 0
    assert known[("gilbert_elliott", 0.5)] < known[("perfect", None)]


def test_episode_info_reports_message_counts():
    _, episode, env = run_with_channel("bernoulli", {"p": 0.25}, seed=0)
    expected_sent = 2 * 2 * (60 + 1)  # drone x robot x (reset + adımlar)
    assert episode["messages_sent"] == expected_sent
    assert 0 < episode["messages_lost"] < expected_sent
    assert episode["messages_lost"] == env.channel.lost
