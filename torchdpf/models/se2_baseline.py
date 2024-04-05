import numpy as np 
import torch 
from torch import nn
from torchdpf.models.initializations import init_xavier_normal
from torchdpf.models.components.observation import ObservationalEncoder, ObservationalLikelihoodEstimator
from torchdpf.models.components.particle import ParticleProposer
from torchdpf.models.components.action import ActionSampler, MotionTransitionModel
from torchdpf.utilities.math import wrap_angle

class DPF_SE2(nn.Module): 
    """
    Differentiable Particle Filter model for SE(2) state space. 
    Based on the original implementation @ https://github.com/tu-rbo/differentiable-particle-filters
    """
    def __init__(self):
        super(DPF_SE2, self).__init__()

        self.observation_encoder = ObservationalEncoder(H=24, W=24,
                                                       in_channels=3,
                                                       embedding_dim=128,
                                                       dropout_rate=0.3)

        self.observation_likelihood_estimator = ObservationalLikelihoodEstimator(state_dim=3,
                                                                                embedding_dim=128,
                                                                                dropout_rate=0.3)

        self.particle_proposer = ParticleProposer(state_dim=3,
                                                embedding_dim=128,
                                                dropout_rate=0.3)  

        self.action_sampler = ActionSampler(action_dim=3,
                                          state_dim=3,
                                          noise_dim=None)

        self.transition_model = MotionTransitionModel(state_dim=3,
                                                   embedding_dim=128,
                                                   dropout_rate=0.3)

        # If false then transition model is disabled
        self.learn_transition_model = False


    def measurement_update(self, encoding, particles, means, stds): 
        """
        Compute the likelihood of the encoded observation for each particle.
        """

        # First encode particles for input 
        # Basically, tile encoding for each particle, then concat with normalized particles
        particle_input = self.encode_particles_for_input(particles, means, stds)
        encoding_input = encoding.unsqueeze(1).repeat(1, particles.shape[1], 1) 
        combined_input = torch.cat([particle_input, encoding_input], dim=-1)
        
        # Estimate the likelihood and remove the last dimension 
        obs_likelihood = self.observation_likelihood_estimator(combined_input).squeeze(-1)

        return obs_likelihood
        
    def encode_particles_for_input(self, particles, means, stds): 
        """
        Normalizes particles and appends sine/cosine of the angle to the particles. 
        """
        normalized_xy = (particles[:, :, :-1] - means[:, :, -1]) / stds[:, :, -1]
        sine_angles = torch.sin(particles[:, :, -1])
        cosine_angles = torch.cos(particles[:, :, -1])
        return torch.cat([normalized_xy, sine_angles, cosine_angles], dim=-1)

    def motion_update(self, actions, particles, means, stds, stop_sampling_gradient=False):
        """
        Perform a noisy motion update on the particles. 
        """

        # Get noisy actions
        noisy_actions = self.action_sampler(actions, stds, particles)

        # Apply noisy actions, depending on whether or not the learned odom model is used
        if self.learn_transition_model: 
            # Encode particles for input 
            # BUG: Two different means, one for actions, one for pose 
            state_input = self.encode_particles_for_input(particles, means, stds)
            action_input = actions / stds
            input = torch.cat([state_input, action_input], dim=-1)
            # Estimate the state delta, scale, and apply 
            delta = self.transition_model(input)
            new_states = particles + delta
            new_states[:, :, 2] = wrap_angle(new_states[:, :, 2]) 
        else: 
            theta = particles[:, :, 2]
            sin_theta = torch.sin(theta)
            cos_theta = torch.cos(theta)
            # Move the particles using the noisy actions 
            new_x = particles[:, :, 0] + noisy_actions[:, :, 0] * cos_theta - noisy_actions[:, :, 1] * sin_theta
            new_y = particles[:, :, 1] + noisy_actions[:, :, 0] * sin_theta + noisy_actions[:, :, 1] * cos_theta
            new_theta = wrap_angle(particles[:, :, 2] + noisy_actions[:, :, 2])

            new_states = torch.stack([new_x, new_y, new_theta], dim=-1)

        return new_states
    

    def propose_particles(self, encoding, num_particles, state_mins, state_maxs): 
        """
            Generates num_particles new proposed partciles based on the current observation embedding. 
            Particle proposals are multiplied by the halved difference between state_maxs and state_mins
        """
        
        tiled_encoding = torch.tile(encoding, (1, num_particles, 1))
        proposed_particles = self.particle_proposer(tiled_encoding)
        
        halved_ranges = state_maxs - state_mins / 2
        avereaged_ranges = (state_maxs + state_mins) / 2

        proposed_particles = torch.cat([proposed_particles[:, :, :2] * halved_ranges[:, :, :2] + avereaged_ranges[:, :, :2],
                                        torch.atan2(proposed_particles[:, :, 3], proposed_particles[:, :, 4], dim=-1)])

        return proposed_particles


    def forward(self, observation, action, particles, particle_weights):
        return