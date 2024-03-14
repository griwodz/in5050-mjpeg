#ifndef TABLES_H
#define TABLES_H

#include <stdint.h>

extern uint8_t yquanttbl[64] __attribute__((aligned(16)));
extern uint8_t uquanttbl[64] __attribute__((aligned(16)));
extern uint8_t vquanttbl[64] __attribute__((aligned(16)));

extern uint16_t DCVLC[2][12];
extern uint8_t  DCVLC_Size[2][12];
extern uint8_t  DCVLC_num_by_length[2][16];
extern uint8_t  DCVLC_data[2][12];

extern uint16_t ACVLC[2][16][11];
extern uint8_t  ACVLC_Size[2][16][11];
extern uint8_t  ACVLC_num_by_length[2][16];
extern uint8_t  ACVLC_data[2][162];

extern uint8_t zigzag_U[64];
extern uint8_t zigzag_V[64];

#endif /* TABLES_H */
