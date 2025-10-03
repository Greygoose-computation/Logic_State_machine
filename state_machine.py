from collections import namedtuple

#from util import plot_automaton

Transition = namedtuple(typename='Transition', field_names=[
                        'source', 'event', 'target'])


class Automaton(object):

    def __init__(self, states, init, events, trans, marked=None, forbidden=None):
        """
        This is the constructor of the automaton.

        At creation, the automaton gets the following attributes assigned:
        :param states: A set of states
        :param init: The initial state
        :param events: A set of events
        :param trans: A set of transitions
        :param marked: (Optional) A set of marked states
        :param forbidden: (Optional) A set of forbidden states
        """
        self.states = states
        self.init = init
        self.events = events
        self.trans = trans
        self.marked = marked if marked else set()
        self.forbidden = forbidden if forbidden else set()

    def __str__(self):
        """
        Prints the automaton in a pretty way.
        """
        return 'states: \n\t{}\n' \
               'init: \n\t{}\n' \
               'events: \n\t{}\n' \
               'transitions: \n\t{}\n' \
               'marked: \n\t{}\n' \
               'forbidden: \n\t{}\n'.format(
                   self.states, self.init, self.events,
                   '\n\t'.join([str(t) for t in self.trans]), self.marked, self.forbidden)

    def __setattr__(self, name, value):
        """Validates and protects the attributes of the automaton"""
        if name in ('states', 'events'):
            value = frozenset(self._validate_set(value))
        elif name == 'init':
            value = self._validate_init(value)
        elif name == 'trans':
            value = frozenset(self._validate_transitions(value))
        elif name in ('marked', 'forbidden'):
            value = frozenset(self._validate_subset(value))
        super(Automaton, self).__setattr__(name, value)

    def __getattribute__(self, name):
        """Returns a regular set of the accessed attribute"""
        if name in ('states', 'events', 'trans', 'marked', 'forbidden'):
            return set(super(Automaton, self).__getattribute__(name))
        else:
            return super(Automaton, self).__getattribute__(name)

    def __eq__(self, other):
        """Checks if two Automata are the same"""
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @staticmethod
    def _validate_set(states):
        """Checks that states is a set and the states in it are strings or integers"""
        assert isinstance(states, set)
        for state in states:
            assert isinstance(state, str) or isinstance(
                state, int), 'A state must be either of type string or integer!'
        return states

    def _validate_subset(self, subset):
        """Validates the set and checks whether the states in the subset are part of the state set"""
        subset = self._validate_set(subset)
        assert subset.issubset(
            self.states), 'Marked and forbidden states must be subsets of all states!'
        return subset

    def _validate_init(self, state):
        """Checks whether the state is part of the state set"""
        assert isinstance(state, str) or isinstance(
            state, int), 'The initial state must be of type string or integer!'
        assert state in self.states, 'The initial state must be member of states!'
        return state

    def _validate_transitions(self, transitions):
        """Checks that all transition elements are part in the respective sets (states, events)"""
        assert isinstance(transitions, set)
        for transition in transitions:
            assert isinstance(transition, Transition)
            assert transition.source in self.states
            assert transition.event in self.events
            assert transition.target in self.states
        return transitions

def filter_trans_by_source(trans, states_to_keep):
    """Returns a new set containing all transitions where the source is in states_to_keep"""
    return {t for t in trans if t.source in states_to_keep}

def filter_trans_by_events(trans, events_to_keep):
    """Returns a new set containing all transitions where the event is in events_to_keep"""
    return {t for t in trans if t.event in events_to_keep}

def filter_trans_by_target(trans, states_to_keep):
    """Returns a new set containing all transitions where the target is in states_to_keep"""
    return {t for t in trans if t.target in states_to_keep}

def extract_elems_from_trans(trans, field):
    """
    Returns a new set with just the elements in a field of all transitions.
    E.g. field='source' for all source states
    or field='event' or field='target'
    """
    return {getattr(t, field) for t in trans}

def flip_trans(trans):
    """ Flips the direction of the transitions in the set"""
    return {Transition(t.target, t.event, t.source) for t in trans}


def reach(events, trans, start_states, forbidden):
    """
    Returns the forward reachable states of a transition set

    :param events: set of events
    :param trans: set of transitions
    :param start_states: set of states
    :param forbidden: set of forbidden states
    """
    # YOUR CODE HERE
    # start an empty state
    reach_states = set()
    current_state = {state for state in start_states}
    current_state = current_state - forbidden
    reach_states.update(current_state)
    while current_state:
        current_trans = filter_trans_by_source(trans, current_state)
        next_state = extract_elems_from_trans(current_trans, 'target')
        next_state = next_state or set()

        new_state = next_state - reach_states
        new_state = new_state - forbidden

        reach_states.update(new_state)

        current_state = new_state

    # raise NotImplementedError()
    return reach_states


def coreach(events, trans, start_states, forbidden):
    """
    Returns the coreachable (backward reachable) states of a transition set

    :param events: set of events
    :param trans: set of transitions
    :param start_states: set of states
    :param forbidden: set of forbidden states
    """
    return reach(events, flip_trans(trans), start_states, forbidden)

def merge_label(label1, label2):
    """Creates a new label based on two labels"""
    return '{}.{}'.format(label1, label2)

def cross_product(setA, setB):
    """Computes the crossproduct of two sets"""
    return {merge_label(a, b) for b in setB for a in setA}


def synch(automaton1, automaton2):
    """
    Computes the synchronous composition of two automata.

    :param automaton1: First automaton
    :param automaton2: Second automaton
    :return: The synchronized automaton
    """
    # Initialize sets for states, transitions, and events
    combined_states = set()
    combined_transitions = set()
    combined_events = automaton1.events | automaton2.events
    initial_state_combined = merge_label(automaton1.init, automaton2.init)

    combined_states.add(initial_state_combined)
    active_positions = [[automaton1.init, automaton2.init]]
    reached_positions = {initial_state_combined}

    while active_positions:
        [curr_state1, curr_state2] = active_positions.pop(0)

        transitions_automaton1 = filter_trans_by_source(automaton1.trans, {curr_state1})
        transitions_automaton2 = filter_trans_by_source(automaton2.trans, {curr_state2})

        if not transitions_automaton1:
            transitions_automaton1 = {Transition(curr_state1, 'None', curr_state1)}
        if not transitions_automaton2:
            transitions_automaton2 = {Transition(curr_state2, 'None', curr_state2)}

        for trans1 in transitions_automaton1:
            for trans2 in transitions_automaton2:
                if trans1.event == trans2.event and trans1.event != 'None':
                    source_state = merge_label(trans1.source, trans2.source)
                    target_state = merge_label(trans1.target, trans2.target)
                    new_transition = Transition(source_state, trans1.event, target_state)
                    if new_transition not in combined_transitions:
                        combined_states.add(target_state)
                        combined_transitions.add(new_transition)
                        active_positions.append([trans1.target, trans2.target])

                if trans1.event not in automaton2.events and trans1.event != 'None':
                    source_state = merge_label(trans1.source, curr_state2)
                    target_state = merge_label(trans1.target, curr_state2)
                    new_transition = Transition(source_state, trans1.event, target_state)
                    if new_transition not in combined_transitions:
                        combined_states.add(target_state)
                        combined_transitions.add(new_transition)
                        active_positions.append([trans1.target, curr_state2])

                if trans2.event not in automaton1.events and trans2.event != 'None':
                    source_state = merge_label(curr_state1, trans2.source)
                    target_state = merge_label(curr_state1, trans2.target)
                    new_transition = Transition(source_state, trans2.event, target_state)
                    if new_transition not in combined_transitions:
                        combined_states.add(target_state)
                        combined_transitions.add(new_transition)
                        active_positions.append([curr_state1, trans2.target])

    # Determining marked states
    if automaton1.marked and automaton2.marked:
        combined_marked_states = cross_product(automaton1.marked, automaton2.marked)
    elif automaton1.marked and not automaton2.marked:
        combined_marked_states = cross_product(automaton1.marked, automaton2.states)
    elif not automaton1.marked and automaton2.marked:
        combined_marked_states = cross_product(automaton1.states, automaton2.marked)
    else:
        combined_marked_states = set()

    combined_marked_states = {state for state in combined_marked_states if state in combined_states}

    # Determining forbidden states
    combined_forbidden_states = cross_product(automaton1.forbidden, automaton2.states) | cross_product(
        automaton1.states, automaton2.forbidden)
    combined_forbidden_states = {state for state in combined_forbidden_states if state in combined_states}

    # Constructing the final synchronized automaton
    synchronized_automaton = Automaton(
        states=combined_states,
        init=initial_state_combined,
        events=combined_events,
        trans=combined_transitions,
        marked=combined_marked_states,
        forbidden=combined_forbidden_states
    )

    return synchronized_automaton


def is_defined_for_p(p_trans, merged_source, shared_sigma_u):
    """
    Boolean check whether there is an uncontrollable transition defined in P
    with the merged_source as start state.

    :param p_trans: Set of transitions in automaton P
    :param merged_source: String label of the merged state in P||Q
    :param shared_sigma_u: Uncontrollable event shared between P and Q.
    """
    validate_inputs(p_trans, merged_source, shared_sigma_u)
    uncontrollable_trans = filter_trans_by_events(p_trans, shared_sigma_u)
    u_trans_from_merged_source = {t for t in uncontrollable_trans
                                  if merged_source.startswith(str(t.source))}
    return u_trans_from_merged_source != set()


def is_defined_for_q(q_trans, merged_source, shared_sigma_u):
    """
    Boolean check whether there is an uncontrollable transition defined in Q
    with the merged_source as start state.

    :param q_trans: Set of transitions in automaton Q
    :param merged_source: String label of the merged state in P||Q
    :param shared_sigma_u: Uncontrollable event shared between P and Q.
    """
    validate_inputs(q_trans, merged_source, shared_sigma_u)
    uncontrollable_trans = filter_trans_by_events(q_trans, shared_sigma_u)
    u_trans_from_merged_source = {t for t in uncontrollable_trans
                                  if merged_source.endswith(str(t.source))}
    return u_trans_from_merged_source != set()


def validate_inputs(trans, source, events):
    assert type(trans) is set
    assert all(type(t) is Transition for t in trans)

    assert type(source) is str

    assert type(events) is set
    assert all(type(e) in (str, int) for e in events)


def supervisor(P, Sp, sigma_u):
    """
    Generates a nonblocking and controllable supervisor for the synchronized system P||Sp.

    :param P: automaton of the plant
    :param Sp: automaton of the specification
    :param sigma_u: set of uncontrollable events
    """

    '''S0 = synch(P,Sp)
    S0_states = S0.states.copy()
    S0_events = S0.events.copy()
    S0_trans = S0.trans.copy()
    S0_mark = S0.marked.copy()
    S0_forb = S0.forbidden.copy()

    prev_unsafe = S0_forb
    unsafe = set()
    flag = True

    while flag:
        prev_unsafe = unsafe
        Q_prim = coreach(S0_events, S0_trans, S0_mark, prev_unsafe)
        Q_bis = coreach(sigma_u, S0_trans, (S0_states - Q_prim), set())
        unsafe = prev_unsafe | Q_bis # Union
        if unsafe == prev_unsafe:
            flag = False

    S_states = S0_states - unsafe

    """if S_states == set(): # There exist no supervisor that can fulfill the specification
        raise ValueError"""

    if not S_states:
        msg = (f"No admissible supervisor states.\n"
           f"|S0_states|={len(S0_states)}, |marked|={len(S0_mark)}, "
           f"|forbidden|={len(S0_forb)}, |unsafe|={len(unsafe)}\n"
           f"sigma_u⊆events? {set(sigma_u).issubset(set(S0_events))}")
        raise ValueError(msg)

    S_state= S_states.copy()
    for k in S_states:
        if is_defined_for_p(P.trans, k, sigma_u) or is_defined_for_q(Sp.trans, k, sigma_u):
            S_state.discard(k)

    S_trans = filter_trans_by_target(S0_trans, S_state)
    S_trans2 = filter_trans_by_source(S_trans, S_state)

    S = Automaton(S_state, S0.init, S0_events ,S_trans2)'''

    # empty set
    S0 = synch(P, Sp)

    # ---- trim formbidden states set
    explicit_forb = (
                            cross_product(P.forbidden, Sp.states) |
                            cross_product(P.states, Sp.forbidden)
                    ) & S0.states

    # uncontrol events isolate plant not in spec
    def plant_enables(q, u):

        return is_defined_for_p(P.trans, q, {u})

    def spec_enables(q, u):
        return is_defined_for_q(Sp.trans, q, {u})

    bad_unctrl = set()
    for q in S0.states:
        for u in sigma_u:
            if (u in Sp.events) and plant_enables(q, u) and not spec_enables(q, u):
                bad_unctrl.add(q)
                break

    unsafe = set(explicit_forb) | bad_unctrl

    #
    X = reach(S0.events, S0.trans, {S0.init}, unsafe)
    if not X or S0.init not in X:
        raise ValueError("No supervisor exists init in forb.")

    # - checks on S0
    by_src = {}
    for t in S0.trans:
        by_src.setdefault(t.source, set()).add(t)

    # main loop
    changed = True
    while changed:
        changed = False

        # block for marked states
        TX = {t for t in S0.trans if t.source in X and t.target in X}
        marked_in_X = S0.marked & X
        if marked_in_X:
            Co = coreach(S0.events, TX, marked_in_X, set())
            X_nb = X & Co
        else:
            # consider all as marked
            X_nb = X

        if X_nb != X:
            X = X_nb
            changed = True
            if not X or S0.init not in X:
                break

        offenders = set()

        for x in X:
            for t in by_src.get(x, set()):
                if t.event in sigma_u and t.target not in X:
                    offenders.add(x)
                    break

        for x in X:
            for u in sigma_u:
                if (u in Sp.events) and plant_enables(x, u) and not spec_enables(x, u):
                    offenders.add(x)
                    break

        if offenders:
            X = X - offenders
            changed = True
            if not X or S0.init not in X:
                break

    if not X or S0.init not in X:
        # print('place_holder_debug')
        raise ValueError("No supervisor exists under Σu and the specification.")

    #  Building the restricted supervisor
    S_trans = {t for t in S0.trans if t.source in X and t.target in X}
    S_mark = S0.marked & X
    S_forb = S0.forbidden & X
    S = Automaton(states=X, init=S0.init, events=S0.events, trans=S_trans,
                  marked=S_mark, forbidden=S_forb)
    return S
