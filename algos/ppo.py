import monitoring

import torch
from torch.utils.data import DataLoader

from models import gradnorm
from train import logger


def ppo_loss_log(newlogprob, oldlogprob, advantage, clip=0.2):
    log_ratio = (newlogprob - oldlogprob)
    # clamp the log to stop infinities (85 is for 32 bit floats)
    log_ratio.clamp_(min=-10.0, max=10.0)
    ratio = torch.exp(log_ratio)

    clipped_ratio = ratio.clamp(1.0 - clip, 1.0 + clip)
    clipped_step = clipped_ratio * advantage.unsqueeze(1)
    full_step = ratio * advantage.unsqueeze(1)
    min_step = torch.stack((full_step, clipped_step), dim=1)
    min_step, clipped = torch.min(min_step, dim=1)

    # logger.info(f'mean advantage : {advantage.mean()}')
    # logger.info(f'mean newlog    : {newlogprob.mean()}')
    # logger.info(f'mean oldlob    : {oldlogprob.mean()}')
    # logger.info(f'mean log_ratio : {log_ratio.mean()}')
    # logger.info(f'mean ratio     : {ratio.mean()}')
    # logger.info(f'mean clip ratio: {clipped_ratio.mean()}')
    # logger.info(f'mean clip step : {clipped_step.mean()}')

    min_step *= -1
    return min_step.mean()


class PurePPOClip:
    def __call__(self, actor, critic, exp_buffer, config, device='cpu'):

        transform, action_transform = config.data.transforms()

        optim = config.algo.optimizer.construct(actor.parameters())
        actor = actor.train()
        actor = actor.to(device)
        dataset = SARAdvantageDataset(exp_buffer, discount_factor=config.algo.discount_factor,
                                      state_transform=transform,
                                      action_transform=action_transform, precision=config.data.precision)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

        batches_p = 0
        for observation, action, reward, advantage in loader:
            batches_p += 1
            for step in range(config.algo.ppo_steps_per_batch):

                #todo catergorical distrubution loss.backward() super slow (pytorch problem)
                optim.zero_grad()

                new_dist = actor(observation)
                new_logprob = new_dist.log_prob(action)
                new_logprob.retain_grad()

                old_dist = actor(observation, old=True)
                old_logprob = old_dist.log_prob(action)
                actor.backup()

                loss = ppo_loss_log(new_logprob, old_logprob, advantage, clip=0.2)
                #logging.info(f'loss {loss.item()}')

                loss.backward()

                optim.step()

        monitoring.info(f'processed {batches_p} batches')

        return actor, None


def log_stats(action, critic, dataset, i, loss, predicted, state, target):
    with torch.no_grad():
        current = critic(state, action)
        total_diff = torch.abs(predicted - current).sum().item()
        mean_diff = total_diff / len(dataset)
        magnitude = gradnorm(critic)
    logger.info(f'loss {loss.item()}')
    logger.info(f'grdnrm {magnitude}')
    logger.info(f'mean_dif {mean_diff}')
    logger.info(
        f'prev mean {predicted.mean()} std {predicted.std()} max {predicted.max()} min {predicted.min()}')
    logger.info(f'target mean {target.mean()} std {target.std()} max {target.max()} min {target.min()}')
    logger.info(
        f'current mean {current.mean()} std {current.std()} max {current.max()} min {current.min()}')
    logger.info(f'iterations {i}')