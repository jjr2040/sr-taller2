from django.core.management.base import BaseCommand, CommandError
import core.utils as utils

class Command(BaseCommand):
    help = 'Recalculate ratings'

    def handle(self, *args, **options):
        
        utils.recalculate_serial()

        self.stdout.write(self.style.SUCCESS('Recalculate finished'))