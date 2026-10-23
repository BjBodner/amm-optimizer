import os
import sys
from functools import partial
from typing import Any, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from reference.constants import (Interaction_Spring_Constants, Mass,
                           Number_Of_Algorithms, Print_Two_Mode_Analytics,
                           Self_Spring_Constants, Total_Resources, print_Cost)

np.random.seed(42)
np.seterr(invalid="ignore")


# TODO
# Make the management of the different algorithms not be hardcoded


def calculate_random_search_amplitude(
    maximal_search: float,
    growth_rate: float,
    growth_factor: float,
    period: float,
    adaptive_amplitude: float,
    epsilon: float = 0.01,
) -> float:
    """
    Calculates the random search scalar amplitude using a periodic exponential function.

    Args:
        maximal_search: Maximum allowed search amplitude
        growth_rate: Rate of exponential growth
        growth_factor: Factor controlling the exponential growth
        period: Period of the sine oscillation
        adaptive_amplitude: Adaptive amplitude for the search
        epsilon: Small constant to avoid singularities (default: 0.01)

    Returns:
        float: Calculated random search amplitude
    """
    # Calculate the phase within the periodic boundary
    phase_ratio = growth_factor * np.pi / (2 * period)
    bounded_phase = np.mod(phase_ratio, np.pi / 2)

    # Calculate the exponential decay term
    decay_term = np.sin(bounded_phase + epsilon) - 1
    exponential_scale = np.exp(growth_rate * decay_term)

    # Calculate the final amplitude
    amplitude = maximal_search * exponential_scale * adaptive_amplitude
    amplitude = np.minimum(amplitude, 10)
    return amplitude


def two_parent_crossover(
    Seed_Parameters, Second_Best_Seed_Parameters, NumberOfSamples, num_params
):
    Seed_Parameters1 = np.tile(Seed_Parameters, (NumberOfSamples, 1)).T
    Second_Best_Seed_Parameters1 = np.tile(
        Second_Best_Seed_Parameters, (NumberOfSamples, 1)
    ).T
    Genes = np.random.rand(num_params, NumberOfSamples)
    RandomSearch = (Genes > 0.5) * Seed_Parameters1 + (
        Genes <= 0.5
    ) * Second_Best_Seed_Parameters1
    return Seed_Parameters1, RandomSearch


def reset_amplitude_if_needed(RandomNumbers, TargetedSearch_Amplitude):
    if np.max(np.abs(TargetedSearch_Amplitude)) > 1000:
        TargetedSearch_Amplitude = 100 * RandomNumbers

    if np.isnan(np.max(TargetedSearch_Amplitude)) == 1:
        TargetedSearch_Amplitude = 100 * RandomNumbers
    return TargetedSearch_Amplitude


def update_growth_factors(
    Parameter_ChangeVector,
    Significant_Change_Value,
    Random_Search_exponential_Growth_Factor,
    Targeted_Search_exponential_Growth_Factor,
):
    if np.sum(np.abs(Parameter_ChangeVector)) > Significant_Change_Value:
        Random_Search_exponential_Growth_Factor = 0
        Targeted_Search_exponential_Growth_Factor += 1

    if np.sum(np.abs(Parameter_ChangeVector)) < Significant_Change_Value:
        Targeted_Search_exponential_Growth_Factor = 0
    return (
        Random_Search_exponential_Growth_Factor,
        Targeted_Search_exponential_Growth_Factor,
    )


def Two_Mode_Optimizer(
    Seed_Parameters,
    historical_solutions,
    num_samples,
    Parameter_ChangeVector,
    Cost_Change,
    Random_Search_exponential_Growth_Factor,
    Targeted_Search_exponential_Growth_Factor,
    Targeted_Search_Decay_Rate,
    Random_Search_Growth_Rate,
    Targeted_Search_Growth_Rate,
    Random_Search_Period,
    Significant_Change_Value,
    Maximal_Random_Search,
    Adaptive_Amplitude,
):
    num_params = len(Seed_Parameters)
    Random_Search_exponential_Growth_Factor1 = Random_Search_exponential_Growth_Factor

    (
        Random_Search_exponential_Growth_Factor,
        Targeted_Search_exponential_Growth_Factor,
    ) = update_growth_factors(
        Parameter_ChangeVector,
        Significant_Change_Value,
        Random_Search_exponential_Growth_Factor,
        Targeted_Search_exponential_Growth_Factor,
    )

    cross_over_operator = (
        two_parent_crossover
        if len(historical_solutions.shape) == 1
        else multi_parent_crossover
    )
    Seed_Parameters1, RandomSearch = cross_over_operator(
        Seed_Parameters, historical_solutions, num_samples, num_params
    )

    RandomSearch = get_random_search(
        num_samples,
        Random_Search_exponential_Growth_Factor,
        Random_Search_Growth_Rate,
        Random_Search_Period,
        Maximal_Random_Search,
        Adaptive_Amplitude,
        num_params,
        RandomSearch,
        Seed_Parameters1,
    )
    TargetedSearch = get_targeted_search(
        num_samples,
        Parameter_ChangeVector,
        Random_Search_exponential_Growth_Factor,
        Targeted_Search_exponential_Growth_Factor,
        Targeted_Search_Decay_Rate,
        Random_Search_Growth_Rate,
        Targeted_Search_Growth_Rate,
        Random_Search_Period,
        Maximal_Random_Search,
        num_params,
        Random_Search_exponential_Growth_Factor1,
    )

    suggested_samples = np.array([Seed_Parameters]).T + TargetedSearch + RandomSearch
    suggested_samples[:, 0] = Seed_Parameters

    Random_Search_exponential_Growth_Factor += 1

    return (
        suggested_samples,
        Random_Search_exponential_Growth_Factor,
        Targeted_Search_exponential_Growth_Factor,
    )


def get_random_search(
    NumberOfSamples,
    Random_Search_exponential_Growth_Factor,
    Random_Search_Growth_Rate,
    Random_Search_Period,
    Maximal_Random_Search,
    Adaptive_Amplitude,
    num_params,
    RandomSearch,
    Seed_Parameters1,
):
    random_search_amplitude = calculate_random_search_amplitude(
        Maximal_Random_Search,
        Random_Search_Growth_Rate,
        Random_Search_exponential_Growth_Factor,
        Random_Search_Period,
        Adaptive_Amplitude,
    )

    samples = 2 * (np.random.rand(num_params, NumberOfSamples) - 0.5)
    RandomSearch += samples * random_search_amplitude - Seed_Parameters1
    return RandomSearch


def get_targeted_search(
    NumberOfSamples,
    Parameter_ChangeVector,
    Random_Search_exponential_Growth_Factor,
    Targeted_Search_exponential_Growth_Factor,
    Targeted_Search_Decay_Rate,
    Random_Search_Growth_Rate,
    Targeted_Search_Growth_Rate,
    Random_Search_Period,
    Maximal_Random_Search,
    num_params,
    Random_Search_exponential_Growth_Factor1,
):
    TargetedSearch = np.zeros((num_params, NumberOfSamples))
    if np.max(Parameter_ChangeVector) > 0:
        TargetedSearch_Amplitude = calc_targeted_search_amplitude(
            NumberOfSamples,
            Random_Search_exponential_Growth_Factor,
            Targeted_Search_exponential_Growth_Factor,
            Targeted_Search_Decay_Rate,
            Random_Search_Growth_Rate,
            Targeted_Search_Growth_Rate,
            Random_Search_Period,
            Maximal_Random_Search,
            Random_Search_exponential_Growth_Factor1,
        )
        TargetedSearch = (Parameter_ChangeVector * TargetedSearch_Amplitude).T
    return TargetedSearch


def multi_parent_crossover(
    Seed_Parameters, Best_Quarter_Of_Individuals, NumberOfSamples, num_params
):
    RandomSearch = np.zeros((num_params, NumberOfSamples))
    Seed_Parameters1 = np.tile(Seed_Parameters, (NumberOfSamples, 1)).T
    Crossover_Indexes = np.random.randint(
        0,
        Best_Quarter_Of_Individuals.shape[1],
        (num_params, NumberOfSamples),
    )
    RandomSearch = Best_Quarter_Of_Individuals[
        np.arange(num_params)[:, None], Crossover_Indexes
    ]
    return Seed_Parameters1, RandomSearch


def calc_exponential_term(
    targeted_search_growth: float,
    targeted_growth_factor: float,
    random_search_decay: float,
    random_growth_factor: float,
) -> float:
    """Calculate exponential term for targeted search amplitude."""
    return (
        targeted_growth_factor * targeted_search_growth
        - random_growth_factor * random_search_decay
    )


def get_base_amplitude(
    exp_term: float, base_amplitude: float, random_numbers: np.ndarray
) -> np.ndarray:
    """Get base amplitude accounting for exponential term."""
    if exp_term < 100:
        return base_amplitude * random_numbers * np.exp(exp_term)
    return 100 * random_numbers


def calc_targeted_search_amplitude(
    NumberOfSamples: int,
    Random_Search_exponential_Growth_Factor: float,
    Targeted_Search_exponential_Growth_Factor: float,
    Targeted_Search_Decay_Rate: float,
    Random_Search_Growth_Rate: float,
    Targeted_Search_Growth_Rate: float,
    Random_Search_Period: float,
    Maximal_Random_Search: float,
    Random_Search_exponential_Growth_Factor1: float,
) -> np.ndarray:
    """Calculate targeted search amplitude using random sampling and exponential scaling."""

    # Generate random numbers
    random_numbers = np.random.rand(NumberOfSamples, 1)

    # Calculate base scalar amplitude
    base_amplitude = calculate_targeted_search_coef(
        Random_Search_Growth_Rate,
        Random_Search_Period,
        Maximal_Random_Search,
        Random_Search_exponential_Growth_Factor1,
    )

    # Calculate exponential term
    exp_term = calc_exponential_term(
        Targeted_Search_Growth_Rate,
        Targeted_Search_exponential_Growth_Factor,
        Targeted_Search_Decay_Rate,
        Random_Search_exponential_Growth_Factor,
    )

    # Get amplitude with exponential scaling
    amplitude = get_base_amplitude(exp_term, base_amplitude, random_numbers)

    # Reset amplitude if needed
    return reset_amplitude_if_needed(random_numbers, amplitude)


def calculate_phase_ratio(growth_factor: float, period: float) -> float:
    """Calculate the phase ratio within periodic boundary."""
    return np.mod(growth_factor * np.pi / (2 * period), np.pi / 2)


def calculate_decay_term(phase: float, epsilon: float = 0.01) -> float:
    """Calculate exponential decay term with small offset to avoid singularities."""
    return np.sin(phase + epsilon) - 1


def calculate_targeted_search_coef(
    random_search_growth_rate: float,
    random_search_period: float,
    maximal_random_search: float,
    growth_factor: float,
) -> float:
    """
    Calculate targeted search coefficient using periodic exponential scaling.

    Args:
        random_search_growth_rate: Rate of exponential growth
        random_search_period: Period of oscillation
        maximal_random_search: Maximum allowed search amplitude
        growth_factor: Factor controlling growth rate

    Returns:
        float: Calculated search coefficient
    """
    # Calculate phase within periodic boundary
    phase = calculate_phase_ratio(growth_factor, random_search_period)

    # Calculate decay term for exponential
    decay = calculate_decay_term(phase)

    # Calculate final amplitude with exponential scaling
    amplitude = maximal_random_search * np.exp(random_search_growth_rate * decay)

    return amplitude


def Symmetric_Section_Conductor(
    Current_Resource_Allocation, Changevector, Hyper_Parameters
):
    ## The difference is that this function can allow for different masses and interactions between the algorithms

    Initial_Resource_Allocation = Hyper_Parameters[:, 0]
    MassVector = Hyper_Parameters[:, 1]
    Self_Interaction_Spring_Constants = Hyper_Parameters[:, 2]
    InteractionMatrix = Hyper_Parameters[:, 3 : 3 + Number_Of_Algorithms]

    Norm = np.sqrt(np.matmul(Changevector, Changevector))
    if Norm != 0:
        Changevector = Changevector / Norm

    R0 = Initial_Resource_Allocation
    R = Current_Resource_Allocation
    K0 = Self_Interaction_Spring_Constants

    ## This updates the allocated resources
    H_Self = -(K0 / MassVector) * (
        R - R0
    )  ## This slowly restores the Resouce allocation to the original values
    H_Interaction = np.matmul(
        InteractionMatrix, (Changevector / MassVector).T
    )  ## this is a resouce conserving interaction
    R = R + H_Self + H_Interaction.T

    RecomendedResourceAllocation1 = np.round(R)

    ## This makes sure that if the total resources aren't conserved then add or subtract from the best Member.
    I = np.argmax(RecomendedResourceAllocation1)
    RecomendedResourceAllocation1[I] = RecomendedResourceAllocation1[I] + (
        np.sum(Initial_Resource_Allocation) - np.sum(RecomendedResourceAllocation1)
    )

    ## This convertes the array to an int array
    RecomendedResourceAllocation = np.arange(len(RecomendedResourceAllocation1))
    for i in range(len(RecomendedResourceAllocation1)):
        if np.isnan(RecomendedResourceAllocation1[i]):
            Changevector = np.zeros(4)
            RecomendedResourceAllocation1 = 16 * np.ones(4)
        RecomendedResourceAllocation[i] = int(RecomendedResourceAllocation1[i])

    return RecomendedResourceAllocation


def Set_Hyperparameters_For_Symmetric_Conductor(
    Total_Resources,
    Number_Of_Algorithms,
    Mass,
    Self_Spring_Constants,
    Interaction_Spring_Constants,
):

    Initial_Resource_Allocation = int(Total_Resources / Number_Of_Algorithms) * np.ones(
        Number_Of_Algorithms
    )

    ## This fixes Rounding issues
    Initial_Resource_Allocation[0] = Initial_Resource_Allocation[0] + (
        Total_Resources - np.sum(Initial_Resource_Allocation)
    )
    MassVector = Mass * np.ones(Number_Of_Algorithms)
    Self_Interaction_Spring_Constants = Self_Spring_Constants * np.ones(
        Number_Of_Algorithms
    )
    InteractionMatrix = Interaction_Spring_Constants * (
        -np.ones((Number_Of_Algorithms, Number_Of_Algorithms))
        + Number_Of_Algorithms * np.diag(np.ones(Number_Of_Algorithms))
    )

    Hyperparameters_Array = np.zeros((Number_Of_Algorithms, 3 + Number_Of_Algorithms))
    Hyperparameters_Array[:, 0] = Initial_Resource_Allocation
    Hyperparameters_Array[:, 1] = MassVector
    Hyperparameters_Array[:, 2] = Self_Interaction_Spring_Constants
    Hyperparameters_Array[:, 3 : 3 + Number_Of_Algorithms] = InteractionMatrix

    return Hyperparameters_Array


class Two_Mode_Suggestion_Tools(object):

    # The class "constructor" - It's actually an initializer
    def __init__(
        self,
        Best_Individual,
        Second_Best_Individual,
        Parameter_ChangeVector,
        Number_Of_Samples,
        Cost_Change,
        Random_Search_exponential_Growth_Factor,
        Targeted_Search_exponential_Growth_Factor,
        Adaptive_Amplitude_Vector,
        Squared_Gradient,
    ):
        self.Best_Individual = Best_Individual
        self.Second_Best_Individual = Second_Best_Individual
        self.Parameter_ChangeVector = Parameter_ChangeVector
        self.Number_Of_Samples = Number_Of_Samples
        self.Cost_Change = Cost_Change
        self.Random_Search_exponential_Growth_Factor = (
            Random_Search_exponential_Growth_Factor
        )
        self.Targeted_Search_exponential_Growth_Factor = (
            Targeted_Search_exponential_Growth_Factor
        )
        self.Adaptive_Amplitude_Vector = Adaptive_Amplitude_Vector
        self.Squared_Gradient = Squared_Gradient


def Create_New_Two_Mode_Suggestion_Tools_Object(
    Number_Of_Parameters,
):  ## this is where the directional pool is initialized
    N = Number_Of_Parameters
    Best_Individual = np.random.rand(N) - 0.5
    Second_Best_Individual = np.random.rand(N) - 0.5
    Parameter_ChangeVector = np.random.rand(N) - 0.5
    Number_Of_Samples = 10
    Cost_Change = -1
    Random_Search_exponential_Growth_Factor = 1.0
    Targeted_Search_exponential_Growth_Factor = 1.0
    Adaptive_Amplitude_Vector = np.random.rand(N) - 0.5
    Squared_Gradient = np.random.rand(N)
    ## Generating and instance For The object the Two mode Suggestions tools object
    return Two_Mode_Suggestion_Tools(
        Best_Individual,
        Second_Best_Individual,
        Parameter_ChangeVector,
        Number_Of_Samples,
        Cost_Change,
        Random_Search_exponential_Growth_Factor,
        Targeted_Search_exponential_Growth_Factor,
        Adaptive_Amplitude_Vector,
        Squared_Gradient,
    )


def get_two_mode_suggestions(
    fun: Any,
    hyperparameters: np.ndarray,
    suggestion_tools: Any,  # TODO add suggestion tools data class
    print_analytics: bool,
    best_quarter_individuals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Any]:
    """
    Unified function for getting two-mode suggestions using either second best individual
    or best quarter of individuals approach.

    Args:
        fun: Objective function
        hyperparameters: Array of hyperparameters [Random_Search_Growth_Rate,
            Random_Search_Period, Maximal_Random_Search, etc.]
        suggestion_tools: SuggestionTools object containing optimization state
        print_analytics: Whether to print analytics
        best_quarter_individuals: Optional array of best quarter individuals

    Returns:
        Tuple of (suggested parameter samples, updated suggestion tools)
    """
    # import pdb; pdb.set_trace()
    # Extract hyperparameters
    (
        random_search_growth_rate,
        random_search_period,
        maximal_random_search,
        targeted_search_growth_rate,
        targeted_search_decay_rate,
        significant_change_value,
        alpha,
        beta,
    ) = hyperparameters[:8]

    # Extract values from suggestion tools
    best_individual = suggestion_tools.Best_Individual
    parameter_change_vector = suggestion_tools.Parameter_ChangeVector
    number_of_samples = suggestion_tools.Number_Of_Samples
    cost_change = suggestion_tools.Cost_Change
    random_search_growth = suggestion_tools.Random_Search_exponential_Growth_Factor
    targeted_search_growth = suggestion_tools.Targeted_Search_exponential_Growth_Factor

    # Create adaptive amplitude matrix
    adaptive_amplitude = np.ones((len(best_individual), number_of_samples))
    adaptive_amplitude[:, :] = np.array([suggestion_tools.Adaptive_Amplitude_Vector]).T

    # # Choose appropriate optimizer and second parameter based on available data
    # optimizer_func = (
    #     Two_Mode_Optimizer2
    #     if best_quarter_individuals is not None
    #     else Two_Mode_Optimizer
    # )
    second_param = (
        best_quarter_individuals
        if best_quarter_individuals is not None
        else suggestion_tools.Second_Best_Individual
    )

    # Call the appropriate optimizer
    (
        suggested_parameter_samples,
        random_search_growth,
        targeted_search_growth,
    ) = Two_Mode_Optimizer(
        best_individual,
        second_param,
        number_of_samples,
        parameter_change_vector,
        cost_change,
        random_search_growth,
        targeted_search_growth,
        targeted_search_decay_rate,
        random_search_growth_rate,
        targeted_search_growth_rate,
        random_search_period,
        significant_change_value,
        maximal_random_search,
        adaptive_amplitude,
    )

    # Update suggestion tools with new values
    suggestion_tools.Random_Search_exponential_Growth_Factor = random_search_growth
    suggestion_tools.Targeted_Search_exponential_Growth_Factor = targeted_search_growth

    return suggested_parameter_samples, suggestion_tools


def get_suggestions_from_two_mode_section(
    fun: Any,
    hyperparameters: List[Any],
    suggestion_tools: List[Any],
    print_analytics: bool,
    best_quarter_individuals: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets suggestions from different sets of hyperparameters using either basic or advanced mode.

    Args:
        fun: Objective function
        hyperparameters: List of hyperparameter sets (exactly 4)
        suggestion_tools: List of suggestion tool objects (exactly 4)
        print_analytics: Whether to print analytics
        best_quarter_individuals: Optional parameter for advanced mode with best performers

    Returns:
        Tuple containing:
            - Combined array of all suggestions
            - Array of indices marking the boundaries between different algorithms
    """
    # Validate inputs
    if len(hyperparameters) != 4 or len(suggestion_tools) != 4:
        raise ValueError(
            "Exactly 4 sets of hyperparameters and suggestion tools are required"
        )

    # Select appropriate suggestion function based on whether best_quarter_individuals is provided
    suggestion_func = partial(
        get_two_mode_suggestions,
        fun,
        print_analytics=print_analytics,
        best_quarter_individuals=best_quarter_individuals,
    )

    # Get suggestions for each set of hyperparameters
    suggestions_and_tools = [
        suggestion_func(h, t) for h, t in zip(hyperparameters, suggestion_tools)
    ]

    # Unpack results
    suggestions = [result[0] for result in suggestions_and_tools]
    updated_tools = [result[1] for result in suggestions_and_tools]

    # Calculate sample lengths
    sample_lengths = [len(suggestion[0, :]) for suggestion in suggestions]

    # Calculate cumulative indices
    cumulative_sums = np.cumsum([0] + sample_lengths)

    # Combine all suggestions
    all_suggestions = np.concatenate(suggestions, axis=1)

    # Update the suggestion tools (modifies the input list)
    for i, tool in enumerate(updated_tools):
        suggestion_tools[i] = tool

    return all_suggestions, cumulative_sums


def Update_Addaptive_Amplitudes(
    Gradients,
    Hyperparameters_1,
    Hyperparameters_2,
    Hyperparameters_3,
    Hyperparameters_4,
    Suggestion_Tools_1,
    Suggestion_Tools_2,
    Suggestion_Tools_3,
    Suggestion_Tools_4,
):

    ## This is to do these two lines for all individuals
    #   alpha = Hyperparameters_1[6]
    #   beta = Hyperparameters_1[7]
    #   S = beta*S + (1-beta)*(np.mean(np.abs(Gradients),1))**2
    #   Adaptive_Amplitude1 = alpha/np.sqrt(S + (alpha**2))
    #   a = Hyperparameters_1[7]*Suggestion_Tools_1.Squared_Gradient + (1-Hyperparameters_1[7])*(np.mean(np.abs(Gradients,1))**2)
    if np.max(np.abs(Gradients)) < 10**20:
        RMS = np.mean(
            Gradients - (np.array([np.mean(Gradients, 1)]).T) ** 2, 1
        )  ## actually its just the square mean
    else:
        RMS = np.zeros(Gradients.shape[0])
    ## Update The First Individaul
    Suggestion_Tools_1.Squared_Gradient = (
        Hyperparameters_1[7] * Suggestion_Tools_1.Squared_Gradient
        + (1 - Hyperparameters_1[7]) * RMS
    )
    Suggestion_Tools_1.Squared_Gradient = Suggestion_Tools_1.Squared_Gradient * np.sign(
        np.mean(Gradients, 1)
    )
    Suggestion_Tools_1.Adaptive_Amplitude = Hyperparameters_1[6] / np.sqrt(
        np.maximum(
            Suggestion_Tools_1.Squared_Gradient + (Hyperparameters_1[6] ** 2),
            0.1 * (Hyperparameters_1[6] ** 2),
        )
    )

    ## Update The Second Individaul
    Suggestion_Tools_2.Squared_Gradient = (
        Hyperparameters_2[7] * Suggestion_Tools_2.Squared_Gradient
        + (1 - Hyperparameters_2[7]) * RMS
    )
    Suggestion_Tools_2.Squared_Gradient = Suggestion_Tools_2.Squared_Gradient * np.sign(
        np.mean(Gradients, 1)
    )
    Suggestion_Tools_2.Adaptive_Amplitude = Hyperparameters_2[6] / np.sqrt(
        np.maximum(
            Suggestion_Tools_2.Squared_Gradient + (Hyperparameters_2[6] ** 2),
            0.1 * (Hyperparameters_2[6] ** 2),
        )
    )

    ## Update The Third  Individaul
    Suggestion_Tools_3.Squared_Gradient = (
        Hyperparameters_3[7] * Suggestion_Tools_3.Squared_Gradient
        + (1 - Hyperparameters_3[7]) * RMS
    )
    Suggestion_Tools_3.Squared_Gradient = Suggestion_Tools_3.Squared_Gradient * np.sign(
        np.mean(Gradients, 1)
    )
    Suggestion_Tools_3.Adaptive_Amplitude = Hyperparameters_3[6] / np.sqrt(
        np.maximum(
            Suggestion_Tools_3.Squared_Gradient + (Hyperparameters_3[6] ** 2),
            0.1 * (Hyperparameters_3[6] ** 2),
        )
    )

    ## Update The Third  Individaul
    Suggestion_Tools_4.Squared_Gradient = (
        Hyperparameters_4[7] * Suggestion_Tools_4.Squared_Gradient
        + (1 - Hyperparameters_4[7]) * RMS
    )
    Suggestion_Tools_4.Squared_Gradient = Suggestion_Tools_4.Squared_Gradient * np.sign(
        np.mean(Gradients, 1)
    )
    Suggestion_Tools_4.Adaptive_Amplitude = Hyperparameters_4[6] / np.sqrt(
        np.maximum(
            Suggestion_Tools_4.Squared_Gradient + (Hyperparameters_4[6] ** 2),
            0.1 * (Hyperparameters_4[6] ** 2),
        )
    )


def Get_Best_Cost_Change_From_Individuals(
    Cost_Change_From_Probe_Samples,
    Indexes_Of_The_Suggestions_From_The_Different_Algorithms1,
):

    Best_Cost_Change_From_Individuals = np.zeros(4)
    Indexes = Indexes_Of_The_Suggestions_From_The_Different_Algorithms1

    ### There is potential for speed up by writing this out explicitely
    for i in range(0, 4):
        index_min = np.argmin(
            Cost_Change_From_Probe_Samples[Indexes[i] : Indexes[i + 1]]
        )
        Best_Cost_Change_From_Individuals[i] = Cost_Change_From_Probe_Samples[
            Indexes[i] : Indexes[i + 1]
        ][index_min]

    return Best_Cost_Change_From_Individuals

    ## Update the Resouce Allocation


def Update_Resource_Allocation(
    Cost_Change_From_Probe_Samples,
    Indexes_Of_The_Suggestions_From_The_Different_Algorithms1,
    Suggestion_Tools_1,
    Suggestion_Tools_2,
    Suggestion_Tools_3,
    Suggestion_Tools_4,
    Hyperparameters_For_Symmetric_Conductor,
):

    # Normalize Cost
    if np.sum(np.abs(Cost_Change_From_Probe_Samples)) != 0:
        Cost_Change_From_Probe_Samples = Cost_Change_From_Probe_Samples / (
            np.sqrt(
                np.matmul(
                    Cost_Change_From_Probe_Samples, Cost_Change_From_Probe_Samples
                )
            )
        )

    Best_Cost_Change_From_Individuals = Get_Best_Cost_Change_From_Individuals(
        Cost_Change_From_Probe_Samples,
        Indexes_Of_The_Suggestions_From_The_Different_Algorithms1,
    )
    Current_Resource_Allocation = np.array(
        [
            Suggestion_Tools_1.Number_Of_Samples,
            Suggestion_Tools_2.Number_Of_Samples,
            Suggestion_Tools_3.Number_Of_Samples,
            Suggestion_Tools_4.Number_Of_Samples,
        ]
    )

    if print_Cost == 1:
        print("Current_Resource_Allocation", Current_Resource_Allocation)

    Recomended_Resource_Allocation = Symmetric_Section_Conductor(
        Current_Resource_Allocation,
        -Best_Cost_Change_From_Individuals,
        Hyperparameters_For_Symmetric_Conductor,
    )
    Current_Resource_Allocation = Recomended_Resource_Allocation

    Suggestion_Tools_1.Number_Of_Samples = Current_Resource_Allocation[0]
    Suggestion_Tools_2.Number_Of_Samples = Current_Resource_Allocation[1]
    Suggestion_Tools_3.Number_Of_Samples = Current_Resource_Allocation[2]
    Suggestion_Tools_4.Number_Of_Samples = Current_Resource_Allocation[3]

    return Best_Cost_Change_From_Individuals


def Update_Best_Individuals_And_Costs_From_This_Round(
    Best_Individual,
    Second_Best_Individual,
    Parameter_ChangeVector,
    Cost_Change,
    Suggestion_Tools_1,
    Suggestion_Tools_2,
    Suggestion_Tools_3,
    Suggestion_Tools_4,
):

    Suggestion_Tools_1.Best_Individual = Best_Individual
    Suggestion_Tools_1.Second_Best_Individual = Second_Best_Individual
    Suggestion_Tools_1.Parameter_ChangeVector = Parameter_ChangeVector[:, 0]
    Suggestion_Tools_1.Cost_Change = Cost_Change

    Suggestion_Tools_2.Best_Individual = Best_Individual
    Suggestion_Tools_2.Second_Best_Individual = Second_Best_Individual
    Suggestion_Tools_2.Parameter_ChangeVector = Parameter_ChangeVector[:, 0]
    Suggestion_Tools_2.Cost_Change = Cost_Change

    Suggestion_Tools_3.Best_Individual = Best_Individual
    Suggestion_Tools_3.Second_Best_Individual = Second_Best_Individual
    Suggestion_Tools_3.Parameter_ChangeVector = Parameter_ChangeVector[:, 0]
    Suggestion_Tools_3.Cost_Change = Cost_Change

    Suggestion_Tools_4.Best_Individual = Best_Individual
    Suggestion_Tools_4.Second_Best_Individual = Second_Best_Individual
    Suggestion_Tools_4.Parameter_ChangeVector = Parameter_ChangeVector[:, 0]
    Suggestion_Tools_4.Cost_Change = Cost_Change


def Optimize_With_TWO_Mode_Section(
    Best_Individual,
    Second_Best_Individual,
    PreviousCost,
    fun,
    current_iteration,
    Total_Number_Of_Iterations,
    Number_Of_Iterations_With_These_Hyperparameters,
    Hyperparameters_1,
    Hyperparameters_2,
    Hyperparameters_3,
    Hyperparameters_4,
    Suggestion_Tools_1,
    Suggestion_Tools_2,
    Suggestion_Tools_3,
    Suggestion_Tools_4,
    Number_Of_Evaluations,
    Allowed_Number_Of_Function_Evaluations,
    ForceStop,
    Cost_Tracker,
    Best_Quarter_Of_Individuals,
):

    Restart = 0
    Best_Cost_Change_From_Individuals1 = np.zeros(
        (Number_Of_Algorithms, Number_Of_Iterations_With_These_Hyperparameters)
    )

    if current_iteration == 0:
        Parameter_ChangeVector = np.zeros((len(Best_Individual), 1))
        Cost_Change = 0
        Update_Best_Individuals_And_Costs_From_This_Round(
            Best_Individual,
            Second_Best_Individual,
            Parameter_ChangeVector,
            Cost_Change,
            Suggestion_Tools_1,
            Suggestion_Tools_2,
            Suggestion_Tools_3,
            Suggestion_Tools_4,
        )

    Hyperparameters_For_Symmetric_Conductor = (
        Set_Hyperparameters_For_Symmetric_Conductor(
            Total_Resources,
            Number_Of_Algorithms,
            Mass,
            Self_Spring_Constants,
            Interaction_Spring_Constants,
        )
    )

    for i in range(0, Number_Of_Iterations_With_These_Hyperparameters):

        current_iteration = current_iteration + 1
        ## Readout ony if wanted
        if print_Cost == 1:
            print("____________________________________________________")
            print("Iteration", current_iteration, "\\", Total_Number_Of_Iterations)

        ##  Get Sample Suggestions from the Section
        if len(Cost_Tracker) <= 1:
            print("using the first function")
            # (
            #     All_Suggestions_From_Two_Mode_Section1,
            #     Indexes_Of_The_Suggestions_From_The_Different_Algorithms1,
            # ) = Get_Suggestions_From_Two_Mode_Section(
            #     fun,
            #     Hyperparameters_1,
            #     Hyperparameters_2,
            #     Hyperparameters_3,
            #     Hyperparameters_4,
            #     Suggestion_Tools_1,
            #     Suggestion_Tools_2,
            #     Suggestion_Tools_3,
            #     Suggestion_Tools_4,
            #     Hyperparameters_For_Symmetric_Conductor,
            # )

            (
                All_Suggestions_From_Two_Mode_Section1,
                Indexes_Of_The_Suggestions_From_The_Different_Algorithms1,
            ) = get_suggestions_from_two_mode_section(
                fun=fun,
                hyperparameters=[
                    Hyperparameters_1,
                    Hyperparameters_2,
                    Hyperparameters_3,
                    Hyperparameters_4,
                ],
                suggestion_tools=[
                    Suggestion_Tools_1,
                    Suggestion_Tools_2,
                    Suggestion_Tools_3,
                    Suggestion_Tools_4,
                ],
                print_analytics=Print_Two_Mode_Analytics,
            )

        if len(Cost_Tracker) > 1:
            print("using the second function")

            (
                All_Suggestions_From_Two_Mode_Section1,
                Indexes_Of_The_Suggestions_From_The_Different_Algorithms1,
            ) = get_suggestions_from_two_mode_section(
                fun=fun,
                hyperparameters=[
                    Hyperparameters_1,
                    Hyperparameters_2,
                    Hyperparameters_3,
                    Hyperparameters_4,
                ],
                suggestion_tools=[
                    Suggestion_Tools_1,
                    Suggestion_Tools_2,
                    Suggestion_Tools_3,
                    Suggestion_Tools_4,
                ],
                print_analytics=Print_Two_Mode_Analytics,
                best_quarter_individuals=Best_Quarter_Of_Individuals,
            )

        Cost = np.zeros(len(All_Suggestions_From_Two_Mode_Section1[1, :]))
        ## Sample the Function at the suggested sample points
        for j in range(0, len(All_Suggestions_From_Two_Mode_Section1[1, :])):

            if Number_Of_Evaluations > Allowed_Number_Of_Function_Evaluations:

                CurrentCost = PreviousCost
                ForceStop = 1
                break
            Cost[j] = fun(All_Suggestions_From_Two_Mode_Section1[:, j])
            Number_Of_Evaluations += 1
        if ForceStop == 1:
            break

        ## Find the Optimzal Sample and calculate the change vectors
        index_min = np.argmin(Cost)
        CurrentCost = Cost[index_min]
        Cost_Change = CurrentCost - PreviousCost
        Parameter_ChangeVector = np.array(
            [All_Suggestions_From_Two_Mode_Section1[:, index_min] - Best_Individual]
        ).T

        # Update the Addaptive Amplitudes for the next search
        # This should minimize the amplitude of ones that have large variance, and increase those which have high variance.
        Cost_Change_From_Probe_Samples = Cost - PreviousCost

        Change_Vectors = (
            All_Suggestions_From_Two_Mode_Section1 - np.array([Best_Individual]).T
        )

        Gradients = Cost_Change_From_Probe_Samples / (
            (np.maximum(Change_Vectors, -0.0001) + np.minimum(Change_Vectors, -0.0001))
            + 0.001
        )

        Update_Addaptive_Amplitudes(
            Gradients,
            Hyperparameters_1,
            Hyperparameters_2,
            Hyperparameters_3,
            Hyperparameters_4,
            Suggestion_Tools_1,
            Suggestion_Tools_2,
            Suggestion_Tools_3,
            Suggestion_Tools_4,
        )

        ## Update Resource Allocation
        Best_Cost_Change_From_Individuals1[:, i] = Update_Resource_Allocation(
            Cost_Change_From_Probe_Samples,
            Indexes_Of_The_Suggestions_From_The_Different_Algorithms1,
            Suggestion_Tools_1,
            Suggestion_Tools_2,
            Suggestion_Tools_3,
            Suggestion_Tools_4,
            Hyperparameters_For_Symmetric_Conductor,
        )

        ## Update Best_Individuals_And_Costs_From_This_Round
        Best_Individual = All_Suggestions_From_Two_Mode_Section1[:, index_min]

        ## Get the Best Quarter of the population
        # This is 1/8th actually
        Best_Quarter_Of_Individuals = All_Suggestions_From_Two_Mode_Section1[
            :,
            np.array(
                np.nonzero(
                    Cost
                    <= np.median(
                        Cost[
                            np.nonzero(
                                Cost
                                <= np.median(Cost[np.nonzero(Cost <= np.median(Cost))])
                            )
                        ]
                    )
                )
            )[0, :],
        ]

        Cost[index_min] = 10000 * Cost[index_min]
        index_min = np.argmin(Cost)
        Second_Best_Individual = All_Suggestions_From_Two_Mode_Section1[:, index_min]
        PreviousCost = CurrentCost
        Update_Best_Individuals_And_Costs_From_This_Round(
            Best_Individual,
            Second_Best_Individual,
            Parameter_ChangeVector,
            Cost_Change,
            Suggestion_Tools_1,
            Suggestion_Tools_2,
            Suggestion_Tools_3,
            Suggestion_Tools_4,
        )

        Cost_Tracker.append(CurrentCost)

        ############### Restart Conditions ##########################
        Restart, reason = check_restart_conditions(
            [
                Suggestion_Tools_1,
                Suggestion_Tools_2,
                Suggestion_Tools_3,
                Suggestion_Tools_4,
            ],
            Cost_Tracker,
        )
        if Restart:
            print(f"Restart Triggered by {reason}")
            break

        if print_Cost == 1:
            print("Run Summary:")
            print(
                "---------------------------------------------------------------------"
            )
            print(
                "CurrentCost is:",
                CurrentCost,
                "After Running Iteration",
                current_iteration,
                "\\",
                Total_Number_Of_Iterations,
            )
            print(
                "the Random_Search_exponential_Growth_Factors are :",
                Suggestion_Tools_1.Random_Search_exponential_Growth_Factor,
                Suggestion_Tools_2.Random_Search_exponential_Growth_Factor,
                Suggestion_Tools_3.Random_Search_exponential_Growth_Factor,
                Suggestion_Tools_4.Random_Search_exponential_Growth_Factor,
            )
            print(
                "the Targeted_Search_exponential_Growth_Factor is:",
                Suggestion_Tools_1.Targeted_Search_exponential_Growth_Factor,
                Suggestion_Tools_2.Targeted_Search_exponential_Growth_Factor,
                Suggestion_Tools_3.Targeted_Search_exponential_Growth_Factor,
                Suggestion_Tools_4.Targeted_Search_exponential_Growth_Factor,
            )
            print(
                "---------------------------------------------------------------------"
            )
            print("")

    return (
        Best_Individual,
        Second_Best_Individual,
        CurrentCost,
        PreviousCost,
        Best_Cost_Change_From_Individuals1,
        current_iteration,
        Number_Of_Evaluations,
        ForceStop,
        Restart,
        Cost_Tracker,
        Best_Quarter_Of_Individuals,
    )


def check_restart_conditions(
    suggestion_tools,
    cost_tracker,
    min_history_length=100,
    improvement_threshold=0.5e-10,
    window_size=30,
):
    """
    Checks if optimization should restart based on various conditions.

    Args:
        suggestion_tools: List of SuggestionTool objects
        cost_tracker: List of historical cost values
        min_history_length: Minimum length of cost history to check conditions
        improvement_threshold: Minimum required improvement threshold
        window_size: Window size for improvement comparisons

    Returns:
        tuple: (should_restart: bool, reason: str)
    """
    # Check growth factor condition
    growth_factors = [
        tool.Random_Search_exponential_Growth_Factor for tool in suggestion_tools
    ]
    if np.min(growth_factors) > 30:
        return True, "Random Search Growth Factor"

    # Only check remaining conditions if we have enough history
    if len(cost_tracker) <= min_history_length:
        return False, ""

    recent_cost = cost_tracker[-1]
    past_cost = cost_tracker[-window_size]
    initial_cost = cost_tracker[0]

    # Check absolute improvement
    absolute_improvement = past_cost - recent_cost
    if absolute_improvement < improvement_threshold:
        return True, "Insufficient Absolute Improvement"

    # Check relative improvement
    relative_improvement = abs(absolute_improvement / initial_cost)
    if relative_improvement < improvement_threshold:
        return True, "Insufficient Relative Improvement"

    # Check stagnation
    if past_cost == recent_cost:
        return True, "Cost Stagnation"

    return False, ""


def Force_Suggested_Hyperparameters_To_be_Within_The_Allowed_Limits(
    suggested_samples, allowed_values
):
    """
    Clips hyperparameter values to be within their allowed bounds.

    Args:
        suggested_samples: Array of shape (n_hyperparams, n_samples) containing suggested values
        allowed_values: Array of shape (n_hyperparams, 2) containing [min, max] bounds

    Returns:
        tuple: (individual sample arrays, full clipped array)
    """
    allowed_values_ = allowed_values[:, :, None]
    clipped_samples = np.clip(
        suggested_samples, allowed_values_[:, 0], allowed_values_[:, 1]
    )
    individual_samples = list(clipped_samples.T)

    # TODO return only the clipped samples, no need to convert to list and back
    return (*individual_samples, clipped_samples)


def Run_Optimization_With_TWO_Mode_Online_Hyperparameter_Addaptation(
    InitialParameters,
    fun,
    Total_Number_Of_Iterations,
    Number_Of_Iterations_With_Hyperparameters_Set,
    Hyperparameters_1,
    Hyperparameters_2,
    Hyperparameters_3,
    Hyperparameters_4,
    Suggestion_Tools_1,
    Suggestion_Tools_2,
    Suggestion_Tools_3,
    Suggestion_Tools_4,
    Hyperparameters_For_Online_Hyperparameter_Search,
    print_Cost,
    Total_Resources,
    Hyperparameter_Allowed_Values,
    Number_Of_Evaluations,
    Allowed_Number_Of_Function_Evaluations,
    ForceStop,
    Restart,
):

    Cost_Tracker = []
    Number_Of_Iterations_With_These_Hyperparameters = (
        Number_Of_Iterations_With_Hyperparameters_Set
    )

    Random_Search_Growth_Rate = Hyperparameters_For_Online_Hyperparameter_Search[0]
    Random_Search_Period = Hyperparameters_For_Online_Hyperparameter_Search[1]
    Maximal_Random_Search = Hyperparameters_For_Online_Hyperparameter_Search[2]
    Targeted_Search_Growth_Rate = Hyperparameters_For_Online_Hyperparameter_Search[3]
    Targeted_Search_Decay_Rate = Hyperparameters_For_Online_Hyperparameter_Search[4]
    Significant_Change_Value = Hyperparameters_For_Online_Hyperparameter_Search[5]
    alpha = Hyperparameters_For_Online_Hyperparameter_Search[6]
    beta = Hyperparameters_For_Online_Hyperparameter_Search[7]

    ## Initialize the suggestion tools for the hyperparameter optimization
    Random_Search_exponential_Growth_Factor = 0
    Targeted_Search_exponential_Growth_Factor = 0
    Hyper_Parameter_ChangeVector = np.zeros(len(Hyperparameters_1))
    Hyper_Parameter_Cost_Change = 0
    current_iteration = 0

    ## This is to set the hyperparameters for optimization
    Number_Of_Samples = Number_Of_Algorithms
    Best_Individual_Hyperparameters = Hyperparameters_1
    Second_Best_Individual_Hyperparameters = Hyperparameters_1

    ## This is to set the function parameters for optimization
    Best_Individual_Parameters = InitialParameters
    Second_Best_Individual_Parameters = InitialParameters
    PreviousCost = fun(InitialParameters)
    Best_Quarter_Of_Individuals = np.zeros(
        (len(InitialParameters), int(Total_Resources / 4))
    )

    ## Update Directional Pools with random gradients
    Random_Parameter_Gradients = np.random.rand(len(InitialParameters), Total_Resources)
    Random_Parameter_Gradients[:, 0] = 0
    Cost_Change_From_Probe_Samples = -np.ones(Total_Resources)

    ## This is find the Cost of the first set of hyperparameters
    (
        Best_Individual_Parameters,
        Second_Best_Individual_Parameters,
        CurrentCost,
        PreviousCost,
        Best_Cost_Change_From_Individuals1,
        current_iteration,
        Number_Of_Evaluations,
        ForceStop,
        Restart,
        Cost_Tracker,
        Best_Quarter_Of_Individuals,
    ) = Optimize_With_TWO_Mode_Section(
        Best_Individual_Parameters,
        Second_Best_Individual_Parameters,
        PreviousCost,
        fun,
        current_iteration,
        Total_Number_Of_Iterations,
        Number_Of_Iterations_With_These_Hyperparameters,
        Hyperparameters_1,
        Hyperparameters_2,
        Hyperparameters_3,
        Hyperparameters_4,
        Suggestion_Tools_1,
        Suggestion_Tools_2,
        Suggestion_Tools_3,
        Suggestion_Tools_4,
        Number_Of_Evaluations,
        Allowed_Number_Of_Function_Evaluations,
        ForceStop,
        Cost_Tracker,
        Best_Quarter_Of_Individuals,
    )

    Cost_Function_For_Hyperparameter_Optimization = (
        np.mean(Best_Cost_Change_From_Individuals1, 1)
        + np.max(Best_Cost_Change_From_Individuals1, 1)
    ) / 2
    index_min = np.argmin(Cost_Function_For_Hyperparameter_Optimization)
    Current_Hyperparameter_Cost = Cost_Function_For_Hyperparameter_Optimization[
        index_min
    ]
    Previous_Hyperparameter_Cost = Cost_Function_For_Hyperparameter_Optimization[
        index_min
    ]

    Adaptive_Amplitude = np.ones((len(Hyperparameters_1), Number_Of_Samples))
    Cost_Function_For_Hyperparameter_Optimization = np.zeros(Number_Of_Samples)
    S = 0

    Number_Of_Hyperparameter_Iterations = np.ceil(
        Total_Number_Of_Iterations / Number_Of_Iterations_With_These_Hyperparameters
    )

    for i in range(0, int(Number_Of_Hyperparameter_Iterations)):

        if Restart == 1:
            break

        ## Generate suggested hyperparameters
        (
            Suggested_Hyper_Parameter_Samples_From_Two_Mode,
            Random_Search_exponential_Growth_Factor,
            Targeted_Search_exponential_Growth_Factor,
        ) = Two_Mode_Optimizer(
            Best_Individual_Hyperparameters,
            Second_Best_Individual_Hyperparameters,
            Number_Of_Samples,
            Hyper_Parameter_ChangeVector,
            Hyper_Parameter_Cost_Change,
            Random_Search_exponential_Growth_Factor,
            Targeted_Search_exponential_Growth_Factor,
            Targeted_Search_Decay_Rate,
            Random_Search_Growth_Rate,
            Targeted_Search_Growth_Rate,
            Random_Search_Period,
            Significant_Change_Value,
            Maximal_Random_Search,
            Adaptive_Amplitude,
        )
        ## Force the hyperparameters to be within the allowed range
        (
            Hyperparameters_1,
            Hyperparameters_2,
            Hyperparameters_3,
            Hyperparameters_4,
            Suggested_Hyper_Parameter_Samples_From_Two_Mode,
        ) = Force_Suggested_Hyperparameters_To_be_Within_The_Allowed_Limits(
            Suggested_Hyper_Parameter_Samples_From_Two_Mode,
            Hyperparameter_Allowed_Values,
        )

        ## Update Directional Pools with random gradients
        Random_Parameter_Gradients = np.random.rand(
            len(InitialParameters), Total_Resources
        )
        Random_Parameter_Gradients[:, 0] = 0
        Cost_Change_From_Probe_Samples = -np.ones(Total_Resources)

        ## Try to optimize with these hyperparameters (Maybe output the suggestion tools as well?)
        (
            Best_Individual_Parameters,
            Second_Best_Individual_Parameters,
            CurrentCost,
            PreviousCost,
            Best_Cost_Change_From_Individuals1,
            current_iteration,
            Number_Of_Evaluations,
            ForceStop,
            Restart,
            Cost_Tracker,
            Best_Quarter_Of_Individuals,
        ) = Optimize_With_TWO_Mode_Section(
            Best_Individual_Parameters,
            Second_Best_Individual_Parameters,
            PreviousCost,
            fun,
            current_iteration,
            Total_Number_Of_Iterations,
            Number_Of_Iterations_With_These_Hyperparameters,
            Hyperparameters_1,
            Hyperparameters_2,
            Hyperparameters_3,
            Hyperparameters_4,
            Suggestion_Tools_1,
            Suggestion_Tools_2,
            Suggestion_Tools_3,
            Suggestion_Tools_4,
            Number_Of_Evaluations,
            Allowed_Number_Of_Function_Evaluations,
            ForceStop,
            Cost_Tracker,
            Best_Quarter_Of_Individuals,
        )

        ## Calculate the Cost_Function_For_Hyperparameter_Optimization
        Cost_Function_For_Hyperparameter_Optimization = (
            np.mean(Best_Cost_Change_From_Individuals1, 1)
            + np.min(Best_Cost_Change_From_Individuals1, 1)
        ) / 2
        Cost_Function_For_Hyperparameter_Optimization1 = np.array(
            [
                Cost_Function_For_Hyperparameter_Optimization[0],
                Cost_Function_For_Hyperparameter_Optimization[1],
                Cost_Function_For_Hyperparameter_Optimization[2],
                Cost_Function_For_Hyperparameter_Optimization[3],
            ]
        )

        ## Find the Best set of hyperparameters and the change vectors
        index_min = np.argmin(Cost_Function_For_Hyperparameter_Optimization)
        Current_Hyperparameter_Cost = Cost_Function_For_Hyperparameter_Optimization[
            index_min
        ]
        Hyper_Parameter_Cost_Change = (
            Current_Hyperparameter_Cost - Previous_Hyperparameter_Cost
        )
        Hyper_Parameter_ChangeVector = (
            Suggested_Hyper_Parameter_Samples_From_Two_Mode[:, index_min]
            - Best_Individual_Hyperparameters
        )

        ## This is the RMS Amplitude Addaptation - For the hyperparameter search
        Cost_Change_From_Probe_Samples = (
            Cost_Function_For_Hyperparameter_Optimization
            - Cost_Function_For_Hyperparameter_Optimization[0]
        )

        Change_Vectors = (
            Suggested_Hyper_Parameter_Samples_From_Two_Mode
            - np.array([Best_Individual_Hyperparameters]).T
        )

        Gradient = Cost_Change_From_Probe_Samples / (
            (np.maximum(Change_Vectors, -0.0001) + np.minimum(Change_Vectors, -0.0001))
            + 0.001
        )

        S = beta * S + (1 - beta) * (np.mean(np.abs(Gradient), 1)) ** 2
        S = S * np.sign(np.mean(Gradient, 1))
        Adaptive_Amplitude1 = alpha / np.sqrt(np.fmax(S + alpha**2, 0.1 * alpha**2))
        Adaptive_Amplitude[:, :] = np.array(
            [Adaptive_Amplitude1]
        ).T  ## this makes an parameter wise adaptive amplitude Based on RMS prob, which we can do element wise multiplication with the random searches.
        # This should minimize the amplitude of ones that have large variance, and increase those which have high variance.

        ## Save the Best Sets of hyperparameters and Cost Function
        Best_Individual_Hyperparameters = (
            Suggested_Hyper_Parameter_Samples_From_Two_Mode[:, index_min]
        )
        Cost_Function_For_Hyperparameter_Optimization[index_min] = (
            10**10
        ) + Cost_Function_For_Hyperparameter_Optimization[index_min]
        index_min = np.argmin(Cost_Function_For_Hyperparameter_Optimization)
        Second_Best_Individual_Hyperparameters = (
            Suggested_Hyper_Parameter_Samples_From_Two_Mode[:, index_min]
        )

        Previous_Hyperparameter_Cost = Current_Hyperparameter_Cost

        if print_Cost == 1:
            print("hyperparameter Summary:")
            print(
                "---------------------------------------------------------------------"
            )
            print(
                "After Running hyperparameter Iteration",
                i,
                "\\",
                int(Number_Of_Hyperparameter_Iterations),
            )
            print(
                "Cost_Function_For_Hyperparameter_Optimization",
                Cost_Function_For_Hyperparameter_Optimization1,
            )
            print("")
            print("Best_Individual_Hyperparameters", Best_Individual_Hyperparameters)
            print("")
            print(
                "the Random_Search_exponential_Growth_Factor for the hyperparameter search is:",
                Random_Search_exponential_Growth_Factor,
            )
            print(
                "the Targeted_Search_exponential_Growth_Factor for the hyperparameter search is:",
                Targeted_Search_exponential_Growth_Factor,
            )
            print(
                "---------------------------------------------------------------------"
            )
            print("")

    return (
        Best_Individual_Parameters,
        CurrentCost,
        Number_Of_Evaluations,
        ForceStop,
        Restart,
    )


def Set_Hyperparameter_Ranges(
    Random_Search_Growth_Rate_Range,
    Random_Search_Period_Range,
    Maximal_Random_Search_Range,
    Targeted_Search_Growth_Rate_Range,
    Targeted_Search_Decay_Rate_Range,
    Significant_Change_Value_Range,
    alpha_Range,
    beta_Range,
):
    Hyperparameter_Allowed_Values = np.zeros((11, 2))

    Hyperparameter_Allowed_Values[0, :] = Random_Search_Growth_Rate_Range
    Hyperparameter_Allowed_Values[1, :] = Random_Search_Period_Range
    Hyperparameter_Allowed_Values[2, :] = Maximal_Random_Search_Range
    Hyperparameter_Allowed_Values[3, :] = Targeted_Search_Growth_Rate_Range
    Hyperparameter_Allowed_Values[4, :] = Targeted_Search_Decay_Rate_Range
    Hyperparameter_Allowed_Values[5, :] = Significant_Change_Value_Range
    Hyperparameter_Allowed_Values[6, :] = alpha_Range
    Hyperparameter_Allowed_Values[7, :] = beta_Range

    return Hyperparameter_Allowed_Values


def Mean_ChangeValue_fit(Maximal_Random_Search, Number_Of_Parameters):
    fit_Parameters = [
        -0.007921662,
        0.003755446,
        5.92 * (10 ** (-5)),
        -2.45 * (10 ** (-7)),
        0.250014494,
    ]  # These were found numerically from the file "Finding_The_Segnificant_Change_Value.ipynb"
    return (
        fit_Parameters[0] * Maximal_Random_Search
        + fit_Parameters[1] * Maximal_Random_Search**2
        + fit_Parameters[2] * Number_Of_Parameters
        + fit_Parameters[3] * Number_Of_Parameters**2
        + fit_Parameters[4] * Maximal_Random_Search * Number_Of_Parameters
    )


def Orchestra_Optimize_With_Two_Mode_Section(
    x0, fun, Allowed_Number_Of_Function_Evaluations, print_Cost
):

    Number_Of_Iterations = int(
        np.maximum(np.ceil(Allowed_Number_Of_Function_Evaluations / 60) - 2, 1)
    )

    Number_Of_Parameters = len(x0)
    InitialParameters = x0
    Total_Number_Of_Iterations = Number_Of_Iterations

    Number_Of_Iterations_With_Hyperparameters_Set = 5

    ############################## Setting up The Conductor (Resource allocation) algorithm ###################################
    Total_Resources = 64
    Number_Of_Algorithms = 4
    Mass = 10
    Self_Spring_Constants = 10
    Interaction_Spring_Constants = 40

    Hyperparameters_For_Symmetric_Conductor = (
        Set_Hyperparameters_For_Symmetric_Conductor(
            Total_Resources,
            Number_Of_Algorithms,
            Mass,
            Self_Spring_Constants,
            Interaction_Spring_Constants,
        )
    )
    Initial_Resource_Allocation = Hyperparameters_For_Symmetric_Conductor[:, 0]
    Current_Resource_Allocation = Hyperparameters_For_Symmetric_Conductor[:, 0]

    Changevector = np.array([0, 0, 0, 0])  ## Negative is bad
    Recomended_Resource_Allocation = Symmetric_Section_Conductor(
        Current_Resource_Allocation,
        Changevector,
        Hyperparameters_For_Symmetric_Conductor,
    )
    Current_Resource_Allocation = Recomended_Resource_Allocation

    #############################################################################################################################

    ######################### Setting the hyperparameter ranges ######################################################################
    Random_Search_Growth_Rate_Range = np.array([0.01, 10])
    Random_Search_Period_Range = np.array([0.1, 10])
    Maximal_Random_Search_Range = np.array([0.0000001, 10])
    Targeted_Search_Growth_Rate_Range = np.array([0.01, 5])
    Targeted_Search_Decay_Rate_Range = np.array([0.1, 100])
    Significant_Change_Value_Range = np.array([10**-30, 0.01])
    alpha_Range = np.array([10**-5, 10**15])  ## (Here Do 10**random)
    beta_Range = np.array([0.1, 1])
    Directional_Pool_Ratio_Range = np.array([0.0001, 1])

    Hyperparameter_Allowed_Values = Set_Hyperparameter_Ranges(
        Random_Search_Growth_Rate_Range,
        Random_Search_Period_Range,
        Maximal_Random_Search_Range,
        Targeted_Search_Growth_Rate_Range,
        Targeted_Search_Decay_Rate_Range,
        Significant_Change_Value_Range,
        alpha_Range,
        beta_Range,
    )
    #############################################################################################################################

    ############################### Setting the Hyperparameters_For_Online_Hyperparameter_improvement  Algorithm #########################
    Hyperparameters_For_Online_Hyperparameter_Search = np.zeros(8)
    Hyperparameters_For_Online_Hyperparameter_Search[
        0
    ] = 5.0  # Random_Search_Growth_Rate
    Hyperparameters_For_Online_Hyperparameter_Search[1] = 5.0  # Random_Search_Period
    Hyperparameters_For_Online_Hyperparameter_Search[2] = 20.02  # Maximal_Random_Search
    Hyperparameters_For_Online_Hyperparameter_Search[
        3
    ] = 0.5  # Targeted_Search_Growth_Rate
    Hyperparameters_For_Online_Hyperparameter_Search[
        4
    ] = 2.0  # Targeted_Search_Decay_Rate
    Hyperparameters_For_Online_Hyperparameter_Search[
        5
    ] = 0.00001 * Mean_ChangeValue_fit(
        Hyperparameters_For_Online_Hyperparameter_Search[2], 8
    )  # Significant_Change_Value
    Hyperparameters_For_Online_Hyperparameter_Search[6] = (
        10**4
    ) / Hyperparameters_For_Online_Hyperparameter_Search[
        2
    ]  # alpha
    Hyperparameters_For_Online_Hyperparameter_Search[7] = 0.6  # beta

    #############################################################################################################################

    Number_Of_Evaluations = 0
    ForceStop = 0
    Restart = 0
    Best_Cost = 10**100
    Number_Of_Restarts = 0
    while ForceStop == 0:
        ########################initialize the different Suggestion_Tools For the Two Mode and the hyperparameters #######################
        Suggestion_Tools_1 = Create_New_Two_Mode_Suggestion_Tools_Object(
            Number_Of_Parameters
        )
        Suggestion_Tools_2 = Create_New_Two_Mode_Suggestion_Tools_Object(
            Number_Of_Parameters
        )
        Suggestion_Tools_3 = Create_New_Two_Mode_Suggestion_Tools_Object(
            Number_Of_Parameters
        )
        Suggestion_Tools_4 = Create_New_Two_Mode_Suggestion_Tools_Object(
            Number_Of_Parameters
        )

        Suggestion_Tools_1.Number_Of_Samples = Current_Resource_Allocation[0]
        Suggestion_Tools_2.Number_Of_Samples = Current_Resource_Allocation[1]
        Suggestion_Tools_3.Number_Of_Samples = Current_Resource_Allocation[2]
        Suggestion_Tools_4.Number_Of_Samples = Current_Resource_Allocation[3]

        ## This is to initiailize the Hyperparameters - Note that they should all be positive numbers (Maybe make 6 and 7 be 10**rand(2))
        Initial_Hyper_Parameter_Samples = 10 * np.random.rand(11, Number_Of_Algorithms)
        (
            Hyperparameters_1,
            Hyperparameters_2,
            Hyperparameters_3,
            Hyperparameters_4,
            Initial_Hyper_Parameter_Samples,
        ) = Force_Suggested_Hyperparameters_To_be_Within_The_Allowed_Limits(
            Initial_Hyper_Parameter_Samples, Hyperparameter_Allowed_Values
        )
        #############################################################################################################################

        (
            Best_Parameters1,
            Best_Cost1,
            Number_Of_Evaluations,
            ForceStop,
            Restart,
        ) = Run_Optimization_With_TWO_Mode_Online_Hyperparameter_Addaptation(
            InitialParameters,
            fun,
            Total_Number_Of_Iterations,
            Number_Of_Iterations_With_Hyperparameters_Set,
            Hyperparameters_1,
            Hyperparameters_2,
            Hyperparameters_3,
            Hyperparameters_4,
            Suggestion_Tools_1,
            Suggestion_Tools_2,
            Suggestion_Tools_3,
            Suggestion_Tools_4,
            Hyperparameters_For_Online_Hyperparameter_Search,
            print_Cost,
            Total_Resources,
            Hyperparameter_Allowed_Values,
            Number_Of_Evaluations,
            Allowed_Number_Of_Function_Evaluations,
            ForceStop,
            Restart,
        )

        if Restart == 1:
            Number_Of_Restarts += 1
        #             print(Number_Of_Restarts)
        #             print(Best_Cost1)

        # if np.mod(Number_Of_Restarts,2) ==1 :
        InitialParameters = Best_Parameters1

        #  if np.mod(Number_Of_Restarts,2) ==0 :
        #      InitialParameters = 10*(np.random.rand(len(Best_Parameters1)) - 0.5)

        Restart = 0
        if Best_Cost1 <= Best_Cost:

            Best_Cost = Best_Cost1
            Best_Parameters = Best_Parameters1

    return Best_Parameters


## Defining Cost Function
class Adaptive_Two_Mode:

    Name = "Adaptive_Two_Mode"

    def __init__(self, Name):
        self.__name__ = "Adaptive_Two_Mode"

    def Optimize(fun, x0, remaining_evals):

        Best_Parameters_After_Optimization = Orchestra_Optimize_With_Two_Mode_Section(
            x0, fun, remaining_evals, print_Cost
        )

        return Best_Parameters_After_Optimization


SOLVER = Adaptive_Two_Mode

print("Finished Loading: ", SOLVER.__name__)
print()
print(SOLVER.__name__ == "Adaptive_Two_Mode")


def fun(x):
    return np.sum((x - 2) ** 2)


if __name__ == "__main__":
    D = 10
    x0 = np.random.rand(D)
    remaining_evals = 1000 * D
    Number_Of_Function_Evaluations = 500 * D

    expected_result = 5.923615233000606e-05
    print(fun(x0))
    x_min1 = SOLVER.Optimize(fun, x0, Number_Of_Function_Evaluations)
    out = fun(x_min1)
    print(fun(x_min1))
    # assert np.isclose(out, expected_result)

    def optimize(fun, x0, remaining_evals):

        Best_Parameters_After_Optimization = Orchestra_Optimize_With_Two_Mode_Section(
            x0, fun, remaining_evals, print_Cost
        )

        return Best_Parameters_After_Optimization
