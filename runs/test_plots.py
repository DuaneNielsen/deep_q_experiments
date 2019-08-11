from algos.td_q import FastPlot


def test_fastplot():
    p = FastPlot(3)
    for i in range(1000):
        p.update(i)
