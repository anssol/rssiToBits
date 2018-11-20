#include <stdio.h>
#include "contiki.h"
#include "net/netstack.h"
#include "sys/rtimer.h"
#include "cpu/cc26xx-cc13xx/rf-core/prop-mode.c"

#define ARRAY_SIZE 8500
static int8_t rssi_array[ARRAY_SIZE];
PROCESS(rssi_scanner, "RSSI Scanner");

AUTOSTART_PROCESSES(&rssi_scanner);

static unsigned int cnt,i,i1,i2;

PROCESS_THREAD(rssi_scanner, ev, data)
{
  PROCESS_BEGIN();
  cnt=0;
  rtimer_clock_t next;
  next = RTIMER_NOW() + RTIMER_SECOND / 1000;

  leds_arch_init();


  while(1)
  {
       ti_lib_gpio_clear_multi_dio(BOARD_LED_ALL);
       next = RTIMER_NOW() + RTIMER_SECOND / 1000;
       rssi_array[cnt++] = (int8_t)get_rssi();
       while(RTIMER_CLOCK_LT( RTIMER_NOW(), next));
       ti_lib_gpio_set_dio(BOARD_IOID_LED_1);
       if(cnt == ARRAY_SIZE)
       {	
	   for(i=0; i < ARRAY_SIZE-10;i++)
             printf("\n%d\n", rssi_array[i]);
  	   cnt = 0;
       }
    
  }

  PROCESS_EXIT();
  PROCESS_END();
}

