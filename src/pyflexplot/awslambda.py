# TODO RMF-81 how does the event look like? -> needs to be done, add ticket
#   the bucket and the object key are required
#   object metadata needs to be given too
def awslambda_handler(event: dict, context: dict) -> dict:
    return {'message': 'OK'}
