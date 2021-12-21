indices, locations, gates, experts_placement = gate(input)
# construct -> dispatch mask without capacity limit
# do uneven alltoall
dispatcher = Flexible_MoEDispatcher(indices, locations, gates, experts_placement, capacity = -1)
# token -> expert
dispatched_input = dispatcher.encode(input)
# expert computation
outputs = experts(dispatched_input)
# expert -> token
combined_output = dispatcher.decode(output)
#
#
-> Next Layer Computation

